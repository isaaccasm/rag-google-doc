# pip install beautifulsoup
# pip install llama-index google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client openai
# pip install faiss-cpu llama-index-vector-stores-faiss
from collections import defaultdict
import json
import logging
import os
from typing import List

import faiss

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode
import numpy as np

from load_keys import load_open_ai_key
from utils import hash_text


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


class FaissRAG:
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
                 model: str = 'gpt-4o-mini',
                 embedding_model_name='text-embedding-3-small',
                 embedding_dim=1536,
                 splitting_symbol='####',
                 save_index_address: str = os.path.join(DATA_ROOT, 'faiss_index')):

        # Input member variable that are modifible if needed.
        self.embedding_dim = embedding_dim  # OpenAI embedding dimension. Change this value if a different embedding is used.
        self.database_filename = 'faiss_text.index'

        self.save_index_address = save_index_address
        self.splitting_symbol = splitting_symbol

        # Load OpenAI LLM and Embedding Model
        self.llm = OpenAI(model=model)  # You can change to "gpt-3.5-turbo"
        self.embed_model = OpenAIEmbedding(model=embedding_model_name)  # OpenAI embedding model

    def _get_documents(self):
        """
        Fetch Google Docs files from specified files or folders, handling both cases correctly.
        """
        raise NotImplementedError()

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
            logger.info(f"🔹 Retrieved chunk: {node.text}")

        return response.response

    def create_index(self, rebuild_index=False):
        """
        Main function to fetch, process, and index documents.
        """
        documents = self._get_documents()
        if not documents:
            logger.error("🚨 No documents found.")
            return None

        # Create and save FAISS index
        faiss_index = self._incremental_update(documents, rebuild_index=rebuild_index)
        logger.info("Document indexing complete. You can now query the documents.")

        # Wrap with LlamaIndex classes and persist
        vector_store = FaissVectorStore(faiss_index=faiss_index)  #, index_path=self.save_index_address)
        index = VectorStoreIndex.from_vector_store(vector_store)
        index.storage_context.persist()

        return faiss_index

    def _incremental_update(self, new_documents, rebuild_index=False):
        """
        Incrementally update the FAISS index:
        - Only updates the chunks from documents passed in `new_documents`
        - Embeds new/changed chunks
        - Removes deleted chunks (only from affected documents)
        - Avoids recomputation of unchanged chunks
        - Removes documents wit no chunks
        """
        faiss_index_path = os.path.join(self.save_index_address, "faiss.index")
        metadata_path = os.path.join(self.save_index_address, "metadata.json")

        # Load FAISS index
        if os.path.isfile(faiss_index_path) and not rebuild_index:
            faiss_index = faiss.read_index(faiss_index_path)
        else:
            faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

        # Load metadata
        if os.path.isfile(metadata_path) and not rebuild_index:
            with open(metadata_path, 'r') as f:
                raw_metadata = json.load(f)
            metadata = defaultdict(dict, {doc_id: doc_meta for doc_id, doc_meta in raw_metadata.items()})
        else:
            metadata = defaultdict(dict)

        # Track changes
        embeddings_to_add, ids_to_add = [], []
        new_doc_hashes = defaultdict(dict)
        chunks_to_keep = defaultdict(set)
        next_vector_id = max(
            [meta["vector_id"] for doc in metadata.values() for meta in doc.values()],
            default=-1
        ) + 1

        # Step 1: Process input documents
        for doc in new_documents:
            doc_id = doc.metadata.get("document_id")
            if not doc_id:
                raise ValueError('Document metadata must contain a "document_id".')

            chunks = [c.strip() for c in doc.text.split(self.splitting_symbol) if c.strip()]

            for i, chunk in enumerate(chunks):
                chunk_hash = hash_text(chunk)
                chunks_to_keep[doc_id].add(chunk_hash)

                if chunk_hash in metadata[doc_id]:
                    continue  # Unchanged

                # New or modified chunk
                embedding = self.embed_model.get_text_embedding(chunk)
                embeddings_to_add.append(embedding)
                ids_to_add.append(next_vector_id)

                new_doc_hashes[doc_id][chunk_hash] = {
                    "vector_id": next_vector_id,
                    "source": doc.metadata["source"],
                    "document_id": doc_id,
                    "chunk_id": i
                }
                next_vector_id += 1

        # Step 2: Identify deleted chunks for passed documents
        deleted_ids = []
        for doc_id in chunks_to_keep:
            for old_hash, meta in metadata[doc_id].items():
                if old_hash not in chunks_to_keep[doc_id]:
                    deleted_ids.append(meta["vector_id"])

        # Step 3: Remove deleted chunks
        if deleted_ids:
            logger.info(f"🗑 Removing {len(deleted_ids)} old embeddings.")
            faiss_index.remove_ids(np.array(deleted_ids))

            # Purge from metadata
            for doc_id in chunks_to_keep:
                for hash_key in list(metadata[doc_id]):
                    if metadata[doc_id][hash_key]["vector_id"] in deleted_ids:
                        metadata[doc_id].pop(hash_key)

        # Step 4: Add new embeddings
        if embeddings_to_add:
            embeddings_array = np.array(embeddings_to_add).astype('float32')
            faiss_index.add_with_ids(embeddings_array, np.array(ids_to_add))

            # Update metadata
            for doc_id in new_doc_hashes:
                metadata[doc_id].update(new_doc_hashes[doc_id])
            logger.info(f"✅ Added {len(embeddings_to_add)} new/updated chunks.")
        else:
            logger.info("✅ No new/updated chunks detected.")

        # Step 5: Remove empty doc entries
        empty_docs = [doc_id for doc_id, chunks in metadata.items() if not chunks]
        for doc_id in empty_docs:
            del metadata[doc_id]

        # Step 6: Persist index and metadata (if anything changed)
        if embeddings_to_add or deleted_ids:
            faiss.write_index(faiss_index, faiss_index_path)
            with open(metadata_path, 'w') as f:
                json.dump(dict(metadata), f)

        return faiss_index
