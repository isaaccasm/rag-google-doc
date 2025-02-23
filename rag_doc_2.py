# pip install beautifulsoup
# pip install llama-index google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai
# pip install faiss-cpu llama-index-vector-stores-faiss

import os

import faiss
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode

from load_keys import load_open_ai_key


# Load API keys for OpenAI
load_open_ai_key()

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'Data')
# Google Drive API Authentication
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly"
]
CREDS_FILE = 'credentials-gdrive.json'


class RagGoogleDoc:
    def __init__(self,
                 folder_ids,
                 model="gpt-4o-mini",
                 local_dir_docs=os.path.join(DATA_ROOT, 'Docs'),
                 save_index_address=os.path.join(DATA_ROOT, 'faiss_index')):
        creds = None
        self.token_file = os.path.join(local_dir_docs, 'token.json')

        if not os.path.isdir(local_dir_docs):
            os.makedirs(local_dir_docs)

        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                credentials = os.path.join(DATA_ROOT, CREDS_FILE)
                flow = InstalledAppFlow.from_client_secrets_file(credentials, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        self.folder_ids = [folder_ids] if isinstance(folder_ids, str) else folder_ids
        self.local_dir_docs = local_dir_docs
        self.save_index_address = save_index_address

        # Input member variable that are modifible if needed.
        self.embedding_dim = 1536  # OpenAI embedding dimension. Change this value if a different embedding is used.
        self.database_filename = 'faiss_text.index'
        self.splitting_symbol = '####'

        self.drive_service = build('drive', 'v3', credentials=creds)
        self.docs_service = build('docs', 'v1', credentials=creds)

        # Load OpenAI LLM and Embedding Model
        self.llm = OpenAI(model=model)  # You can change to "gpt-3.5-turbo"
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # OpenAI embedding model

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
                print(f"🚨 File or folder with ID '{item_id}' not found.")
                return []

            mime_type = file_info["mimeType"]
            file_name = file_info["name"]

            # If it's a Google Doc, fetch its content
            if mime_type == "application/vnd.google-apps.document":
                print(f"📄 Fetching document: {file_name}")
                text = self._get_google_docs_text(item_id)
                return [Document(text=text, metadata={"source": file_name})]

            # If it's a folder, recursively process its contents
            elif mime_type == "application/vnd.google-apps.folder":
                print(f"📂 Entering folder: {file_name}")

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
                        print(f"📄 Fetching document: {file['name']}")
                        text = self._get_google_docs_text(doc_id)
                        documents.append(Document(text=text, metadata={"source": file["name"]}))

                    elif doc_mime == "application/vnd.google-apps.folder":
                        print(f"📂 Entering subfolder: {file['name']}")
                        documents.extend(fetch_files(doc_id))  # Recursively fetch files

                return documents

            else:
                print(f"⚠️ Skipping unsupported file type: {file_name} ({mime_type})")
                return []

        # Process all provided IDs (folders or files)
        documents = []
        for item_id in self.folder_ids:
            documents.extend(fetch_files(item_id))

        return documents


    def _split_documents_into_text_chunks(self, documents):
        """
        Split all the documents using the self.splitting_symbol
        :param documents: A list of documents
        :return: A group of nodes
        """
        nodes = []

        for doc in documents:
            # Split text into chunks using 'self.splitting_symbol'
            chunks = doc.text.split(self.splitting_symbol)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty sections

            for i, chunk in enumerate(chunks):
                nodes.append(
                    TextNode(
                        text=chunk,
                        metadata={"source": doc.metadata["source"], "chunk_id": i}  # ✅ Store text in metadata
                    )
                )

        return nodes

    def _save_index(self, nodes):
        """
        Save documents into a FAISS vector index, splitting them based on self.splitting_symbol delimiters.
        """

        # Use FAISS for storing vector embeddings
        # A FAISS (Facebook AI Similarity Search) vector store is a high-performance library for storing and searching
        # large collections of vector embeddings efficiently. It is widely used in retrieval-augmented generation (RAG)
        # and semantic search applications. It enables:
        # - Fast nearest neighbor search on large-scale datasets.
        # - Efficient similarity matching between query vectors and stored document embeddings.
        # - Optimized GPU acceleration for large-scale vector searches.
        faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        # Wrap FAISS index in FaissVectorStore
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Create a storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Store embeddings by passing **chunked** documents into the FAISS index
        index = VectorStoreIndex(nodes, storage_context=storage_context)

        # Debugging: Check if FAISS index contains anything
        print(f"🔍 FAISS index contains {faiss_index.ntotal} vectors.")

        # Ensure the save directory exists
        if self.save_index_address:
            os.makedirs(self.save_index_address, exist_ok=True)
            faiss_index_path = os.path.join(self.save_index_address, "faiss")

            index.storage_context.persist(persist_dir=self.save_index_address)

            if not os.path.isfile(faiss_index_path):
                print(f"⚠️ FAISS vector index not found at {faiss_index_path}, attempting to save it manually.")
                faiss.write_index(faiss_index, faiss_index_path)

        return index  # Return the FAISS-backed index

    def _load_index(self):
        """
        Load FAISS vector index from storage.
        """
        filename_index = os.path.join(self.save_index_address, 'index_store.json')
        if not os.path.exists(filename_index):
            raise FileNotFoundError(f"index not found at {filename_index}")

        vector_store = FaissVectorStore.from_persist_dir(self.save_index_address)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.save_index_address
        )
        return load_index_from_storage(storage_context=storage_context)

    def query_index(self, query):
        """
        Query the stored index using an LLM.
        """
        index = self._load_index()

        # Create a retriever to fetch relevant text chunks
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)  # Fetch top 3 relevant chunks

        # Create a response synthesizer
        response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

        # Create the query engine with the retriever and response synthesizer
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )

        response = query_engine.query(query)

        # Extract retrieved nodes (to ensure text retrieval works)
        for node in response.source_nodes:
            print(f"🔹 Retrieved chunk: {node.text}")

        return response.response

    def create_index(self):
        """
        Main function to fetch, process, and index documents.
        """
        documents = self._get_documents()
        if not documents:
            print("No documents found.")
            return None

        text_chunks = self._split_documents_into_text_chunks(documents)

        if not text_chunks:
            print("🚨 No nodes found! The FAISS index will be empty.")
            return None

        # Create and save FAISS index
        index = self._save_index(text_chunks)
        print("Document indexing complete. You can now query the documents.")

        return index

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
    google_drive_folder_ids = [
        # '14rNXFJe-WnK3AyIoDSX_8zNCVVFiOXed',
        # '12HEHe7876pCtuL5Z4makI-5E5cd2fLx4'
        '1US5wqXvJYw6u98yCfxsr9E42lmoaSDyp'
    ]
    files = ['1SDcxHfsbM4s3Xl3t3KEsuQvTWeyLtCE4GQpwkCd78Ck']

    rag_obj = RagGoogleDoc(google_drive_folder_ids[0], save_index_address='Data/google_db')
    #db = rag_obj.create_index()

    query = 'Summarise the paper: Region Refinement Network for Salient Object Detection'
    rag_obj.query_index(query)
