import os
import json
import unittest
import numpy as np
import faiss
from unittest.mock import MagicMock, patch
from llama_index.core import Document
from faiss_rag import FaissRAG
from utils import hash_text as original_hash_text


def mock_embedding(text):
    return np.array([len(text)] * 1536, dtype=np.float32)


def mock_hash_text(text):
    return hash(text) % 1000000


class TestFaissRAG(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "./temp_faiss_index"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.rag = FaissRAG(save_index_address=self.temp_dir)

        self.patcher = patch('llama_index.embeddings.openai.OpenAIEmbedding.get_text_embedding',
                             side_effect=mock_embedding)
        self.mock_get_text_embedding = self.patcher.start()

        global hash_text
        self.original_hash_text = original_hash_text
        hash_text = mock_hash_text

    def tearDown(self):
        self.patcher.stop()
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

        global hash_text
        hash_text = self.original_hash_text

    def test_add_new_documents(self):
        doc1 = Document(text="This is a test document.", metadata={"source": "doc1", "document_id": "doc1"})
        doc2 = Document(text="Another document.", metadata={"source": "doc2", "document_id": "doc2"})

        self.rag._incremental_update([doc1, doc2], rebuild_index=True)
        faiss_index = faiss.read_index(os.path.join(self.temp_dir, "faiss.index"))
        self.assertEqual(faiss_index.ntotal, 2)

    def test_update_one_document_only(self):
        doc1 = Document(text="Text A.", metadata={"source": "doc1", "document_id": "doc1"})
        doc2 = Document(text="Text B.", metadata={"source": "doc2", "document_id": "doc2"})
        self.rag._incremental_update([doc1, doc2], rebuild_index=True)

        # Modify doc1 only
        doc1_modified = Document(text="Text A updated.", metadata={"source": "doc1", "document_id": "doc1"})
        self.rag._incremental_update([doc1_modified])

        faiss_index = faiss.read_index(os.path.join(self.temp_dir, "faiss.index"))
        self.assertEqual(faiss_index.ntotal, 2)  # Still two chunks total

    def test_remove_chunks_from_one_doc_only(self):
        doc1 = Document(text="Chunk 1.####Chunk 2.", metadata={"source": "doc1", "document_id": "doc1"})
        self.rag._incremental_update([doc1], rebuild_index=True)

        # Remove one chunk
        doc1_trimmed = Document(text="Chunk 1.", metadata={"source": "doc1", "document_id": "doc1"})
        self.rag._incremental_update([doc1_trimmed])

        faiss_index = faiss.read_index(os.path.join(self.temp_dir, "faiss.index"))
        self.assertEqual(faiss_index.ntotal, 1)

    def test_no_change_means_no_update(self):
        doc = Document(text="Consistent text.", metadata={"source": "doc", "document_id": "doc"})
        self.rag._incremental_update([doc], rebuild_index=True)

        faiss_index = faiss.read_index(os.path.join(self.temp_dir, "faiss.index"))
        count_before = faiss_index.ntotal
        self.rag._incremental_update([doc])  # No changes
        faiss_index = faiss.read_index(os.path.join(self.temp_dir, "faiss.index"))
        count_after = faiss_index.ntotal
        self.assertEqual(count_before, count_after)


if __name__ == "__main__":
    unittest.main()
