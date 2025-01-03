# Install necessary libraries
# pip install llama-index google-api-python-client openai
import hashlib
import json
import os

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.storage.storage_context import StorageContext


# Authenticate and Access Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/documents.readonly']
CREDS_FILE = 'credentials-gdrive.json'


def summary_splitter(documents):
    """
    Custom splitter that divides documents based on summary structure
    (e.g., titles with larger fonts or specific markers like '###').
    """
    custom_chunks = []

    for doc in documents:
        # Example: Split by markers like '###' indicating a new section
        sections = doc.text.split('###')
        for section in sections:
            if section.strip():  # Avoid empty chunks
                custom_chunks.append(Document(text=section.strip()))

    return custom_chunks


class RagGoogleDoc:
    def __init__(self, folder_ids, local_dir_docs='Data/Docs', save_index_address=None):
        creds = None
        self.token_file = os.path.join(local_dir_docs, 'token.json')
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        self.folder_ids = [folder_ids] if isinstance(folder_ids, str) else folder_ids
        self.local_dir_docs = local_dir_docs
        self.save_index_address = save_index_address

        self.drive_service = build('drive', 'v3', credentials=creds)
        self.docs_service = build('docs', 'v1', credentials=creds)

    def get_document_and_texts(self):
        """
        Fetch Google Docs files only from the first level of subdirectories under the given folder_id.
        """
        documents = []

        for folder_id in self.folder_ids:
            # Fetch first-level subdirectories
            subfolders_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            subfolders_results = self.drive_service.files().list(q=subfolders_query, fields='files(id, name)').execute()
            subfolders = subfolders_results.get('files', [])

            # For each subdirectory, fetch its Google Docs files
            for subfolder in subfolders:
                print(f"Processing subfolder: {subfolder['name']}")
                subfolder_id = subfolder['id']
                files_query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.document'"
                files_results = self.drive_service.files().list(q=files_query, fields='files(id, name)').execute()
                files = files_results.get('files', [])

                for file in files:
                    doc_id = file['id']
                    doc_name = file['name']
                    print(f"Fetching document: {doc_name}")
                    text = self.get_google_docs_text(doc_id)
                    documents.append({"name": doc_name, "content": text})

        return documents

    def get_google_docs_text(self, doc_id):
        document = self.docs_service.documents().get(documentId=doc_id).execute()
        content = document.get('body', {}).get('content', [])
        output = ''

        for element in content:
            if 'paragraph' in element:
                paragraph = element['paragraph']
                style = paragraph.get('paragraphStyle', {})
                named_style = style.get('namedStyleType', 'NORMAL_TEXT')

                text_content = ""
                if named_style == 'HEADING_2':
                    text_content += '### '

                for text_run in paragraph.get('elements', []):
                    if 'textRun' in text_run:
                        text_content += text_run['textRun']['content']

                output += text_content

        return output

    def save_documents(self, documents):
        if not os.path.exists(self.local_dir_docs):
            os.makedirs(self.local_dir_docs)
        for i, doc in enumerate(documents):
            with open(f'{self.local_dir_docs}/doc_{i}.txt', "w") as f:
                f.write(doc["content"])

    def create_chunks_with_ids(self, doc_name, chunks):
        for i, chunk in enumerate(chunks):
            chunk.doc_id = f"{doc_name}_chunk_{i}"
        return chunks

    def run(self):
        documents = self.get_document_and_texts()
        self.save_documents(documents)

        # Ensure the save_index_address directory exists
        if self.save_index_address and not os.path.exists(self.save_index_address):
            os.makedirs(self.save_index_address)

        # Load documents and split them
        raw_documents = SimpleDirectoryReader(self.local_dir_docs).load_data()
        custom_chunks = summary_splitter(raw_documents)
        # custom_chunks = self.create_chunks_with_ids(doc_name, custom_chunks)

        # Load or create the index
        if self.save_index_address and os.path.exists(os.path.join(self.save_index_address, '/docstore.json')):
            storage_context = StorageContext.from_defaults(persist_dir=self.save_index_address)
            index = VectorStoreIndex(storage_context=storage_context)
        else:
            index = VectorStoreIndex.from_documents(custom_chunks)
            if self.save_index_address:
                index.storage_context.persist(persist_dir=self.save_index_address)

        return index


# Step 5: Query the RAG system
def query_index(query, index):
    response = index.query(query)
    return response


folder_ids = [
    '14rNXFJe-WnK3AyIoDSX_8zNCVVFiOXed',
    # '12HEHe7876pCtuL5Z4makI-5E5cd2fLx4'
]
index = RagGoogleDoc(folder_ids[0], save_index_address='Data/google_doc_index').run()

# Example query
question = "What are the advancements in reinforcement learning?"
response = query_index(question, index)
print("\nAnswer:", response)

