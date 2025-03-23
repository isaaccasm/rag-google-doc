# pip install beautifulsoup
# pip install llama-index google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai
# pip install faiss-cpu llama-index-vector-stores-faiss

import logging
import os
from typing import List

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from llama_index.core import Document

from load_keys import load_open_ai_key
from faiss_rag import FaissRAG


# Load API keys for OpenAI
load_open_ai_key()

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'Data')
# Google Drive API Authentication
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly"
]
CREDS_FILE = 'credentials-gdrive.json'


logger = logging.getLogger(__name__)


class RagGoogleDoc(FaissRAG):
    """
    A class for creating and querying a FAISS vector database using Google Docs.

    This class enables retrieval-augmented generation (RAG) by fetching documents
    from Google Drive, processing their content, indexing them with FAISS,
    and querying the indexed data for relevant information.

    Attributes:
    -----------
    folder_ids : list
        A list of Google Drive folder or file IDs containing the Google Docs to index.

    model : str, optional
        The embedding model to use for generating vector representations of text.
        Defaults to gpt-4o-mini.

    local_dir_docs : str, optional
        The local directory where fetched Google Docs are temporarily stored.
        If not provided, a default directory is used.

    save_index_address : str, optional
        The directory where the FAISS index and associated metadata are saved.

    Methods:
    --------
    create_index():
        Retrieves Google Docs, processes them into text chunks,
        and builds a FAISS vector index.

    query_index(query: str):
        Searches the FAISS index for the most relevant text chunks based on the query
        and returns a response generated using an LLM.
    """
    def __init__(self,
                 folder_ids: List[str],
                 model: str = 'gpt-4o-mini',
                 embedding_model_name='text-embedding-3-small',
                 embedding_dim=1536,
                 splitting_symbol='####',
                 local_dir_docs: str = os.path.join(DATA_ROOT, 'Docs'),
                 save_index_address: str = os.path.join(DATA_ROOT, 'faiss_index')):

        super().__init__(model=model,
                         embedding_model_name=embedding_model_name,
                         embedding_dim=embedding_dim,
                         splitting_symbol=splitting_symbol,
                         save_index_address=save_index_address)
        creds = None
        self.token_file = os.path.join(local_dir_docs, 'token.json')

        if not os.path.isdir(local_dir_docs):
            os.makedirs(local_dir_docs)

        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError:
                    creds = None  # Force interactive login
            if not creds:
                credentials_path = os.path.join(DATA_ROOT, CREDS_FILE)
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        self.folder_ids = [folder_ids] if isinstance(folder_ids, str) else folder_ids
        self.local_dir_docs = local_dir_docs

        self.drive_service = build('drive', 'v3', credentials=creds)
        self.docs_service = build('docs', 'v1', credentials=creds)

    def _get_google_docs_text(self, doc_id):
        """
        Fetches text content from a Google Document.
        """
        document = self.docs_service.documents().get(documentId=doc_id).execute()
        content = document.get('body', {}).get('content', [])
        output = ''

        for element in content:
            if 'paragraph' in element:
                paragraph = element['paragraph']
                style = paragraph.get('paragraphStyle', {})
                named_style = style.get('namedStyleType', 'NORMAL_TEXT')

                text_content = ""
                if named_style == 'HEADING_2' or named_style == 'HEADING_1':
                    text_content += self.splitting_symbol

                for text_run in paragraph.get('elements', []):
                    if 'textRun' in text_run:
                        text_content += text_run['textRun']['content']

                output += text_content.strip() + '\n'

        return output

    def _get_documents(self):
        """
        Fetch Google Docs files from specified files or folders, handling both cases correctly.
        """

        def fetch_files(item_id):
            """Recursively fetch Google Docs from a file or folder."""
            file_info = self.drive_service.files().get(
                fileId=item_id, fields="id, name, mimeType"
            ).execute()

            if not file_info:  # If the API didn't return anything, exit
                logger.error(f"🚨 File or folder with ID '{item_id}' not found.")
                return []

            mime_type = file_info["mimeType"]
            file_name = file_info["name"]

            # If it's a Google Doc, fetch its content
            if mime_type == "application/vnd.google-apps.document":
                logger.info(f"📄 Fetching document: {file_name}")
                text = self._get_google_docs_text(item_id)
                return [Document(text=text, metadata={"source": file_name,
                                                      "document_id": file_name})]

            # If it's a folder, recursively process its contents
            elif mime_type == "application/vnd.google-apps.folder":
                logger.info(f"📂 Entering folder: {file_name}")

                documents = []

                # Fetch all files (Google Docs) inside the folder
                files_query = f"'{item_id}' in parents"
                files_results = self.drive_service.files().list(q=files_query,
                                                                fields="files(id, name, mimeType)").execute()
                files = files_results.get("files", [])

                for file in files:
                    doc_id = file["id"]
                    doc_mime = file["mimeType"]

                    # Process Google Docs or subfolders recursively
                    if doc_mime == "application/vnd.google-apps.document":
                        logger.debug(f"📄 Fetching document: {file['name']}")
                        text = self._get_google_docs_text(doc_id)
                        documents.append(Document(text=text, metadata={"source": file["name"],
                                                                       "document_id": file["name"]}))

                    elif doc_mime == "application/vnd.google-apps.folder":
                        logger.debug(f"📂 Entering subfolder: {file['name']}")
                        documents.extend(fetch_files(doc_id))  # Recursively fetch files

                return documents

            else:
                logger.debug(f"⚠️ Skipping unsupported file type: {file_name} ({mime_type})")
                return []

        # Process all provided IDs (folders or files)
        documents = []
        for item_id in self.folder_ids:
            documents.extend(fetch_files(item_id))

        return documents

    def test_google_drive_query(self):
        folder_id = self.folder_ids[0]
        print(f"Searching in folder: {folder_id}")

        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
        results = self.drive_service.files().list(
            q=query,
            fields="files(id, name, owners, parents)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get("files", [])

        print("API Response:", files)
        return files

    def test_drive_api(self):
        """
        Test if Google Drive API is returning any files at all.
        """
        print("🔍 Checking Google Drive API...")

        try:
            results = self.drive_service.files().list(fields="files(id, name, mimeType, parents)").execute()
            files = results.get("files", [])

            if not files:
                print("🚨 No files found. The API may not have the right permissions.")
            else:
                print(f"✅ Found {len(files)} files. Listing first 10:")
                for file in files[:10]:  # Print first 10 files
                    print(
                        f"📄 File: {file['name']} (ID: {file['id']}) | Type: {file['mimeType']} | Parent: {file.get('parents', ['Unknown'])}")

        except Exception as e:
            print(f"🚨 API Error: {e}")

    def search_for_folder(self, folder_ids, is_file=False):
        query = "mimeType='application/vnd.google-apps.document'"
        results = self.drive_service.files().list(
            q=query,
            fields="files(id, name, parents)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        files = results.get("files", [])

        found = [False for _ in folder_ids]
        parents = {}

        print("\n📂 Checking Parent Folders for Google Docs:")
        for file in files:  # Print first 20 files
            parent_id = file.get("parents", ["Unknown"])[0]
            file_id = file["id"]
            if not is_file and parent_id in folder_ids:
                found[folder_ids.index(parent_id)] = True
            if is_file and file_id in folder_ids:
                found[folder_ids.index(file_id)] = True
                parents[file_id] = parent_id
            # print(f"📄 File: {file['name']} (ID: {file['id']}) | Parent Folder: {parent_id}")

        for i, f in enumerate(folder_ids):
            reset_colour = "\033[0m"
            parent_text = ''
            if found[i]:
                text = 'Found'
                colour = "\033[92m"
                if is_file:
                    parent_text = f'  --  parent_id: {parents[f]}'
            else:
                text = 'Not found'
                colour = "\033[91m"
            print(f'{f}: {colour}{text}{reset_colour}{parent_text}')


if __name__ == '__main__':
    db = RagGoogleDoc([], save_index_address='./Data/google_db')
    response = db.query_index('I want to ask about the fastest salient object detection algorithm. Tell me what it is')
    print(response)
