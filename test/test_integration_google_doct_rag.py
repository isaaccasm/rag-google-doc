import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from llama_index.core import Document
from google_doc_rag import RagGoogleDoc


class TestRagGoogleDocIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temp directory to store index
        self.temp_dir = "./temp_google_doc_index"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Patch OpenAI embedding + LLM to avoid real API calls
        self.embedding_patcher = patch(
            'llama_index.embeddings.openai.OpenAIEmbedding.get_text_embedding',
            side_effect=lambda text: np.array([float(len(text))] * 1536, dtype=np.float32)
        )
        self.llm_patcher = patch(
            'llama_index.llms.openai.OpenAI.complete',
            return_value="This is a mocked LLM response."
        )

        self.mock_embed = self.embedding_patcher.start()
        self.mock_llm = self.llm_patcher.start()

        # Create dummy RagGoogleDoc with mocked document loader
        self.rag = RagGoogleDoc(folder_ids=[], save_index_address=self.temp_dir)
        self.rag._get_documents = MagicMock(return_value=[
            Document(text="First test document.####Second part of the document.",
                     metadata={"source": "TestDoc1", "document_id": "TestDoc1"})
        ])

    def tearDown(self):
        self.embedding_patcher.stop()
        self.llm_patcher.stop()
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_pipeline(self):
        # Step 1: Create index
        index = self.rag.create_index()
        self.assertTrue(index is not None)

        # Step 2: Query index (mocked LLM will respond)
        result = self.rag.query_index("What does the test document say?")
        self.assertIn("mocked", result.lower())
        print("Query result:", result)


if __name__ == "__main__":
    unittest.main()
