import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag_pipeline import RAGPipeline


class RAGPipelineProviderTests(unittest.TestCase):
    def test_prefers_google_when_google_api_key_is_available(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            self.assertEqual(pipeline._get_llm_provider(), "google")

    def test_uses_anthropic_when_anthropic_key_is_available(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            self.assertEqual(pipeline._get_llm_provider(), "anthropic")

    def test_loads_google_key_from_dotenv(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("GOOGLE_API_KEY=test-dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
                    self.assertEqual(pipeline._get_llm_provider(), "google")


if __name__ == "__main__":
    unittest.main()
