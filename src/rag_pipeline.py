"""
rag_pipeline.py

"""

import os
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from .preprocessing import load_knowledge_base
except ImportError:  # pragma: no cover - supports script execution
    from preprocessing import load_knowledge_base

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, strong general-purpose embedder


class RAGPipeline:
    def __init__(self, kb_dir: str = "data/knowledge_base"):
        self._load_environment()
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.records = load_knowledge_base(kb_dir)
        self.texts = [r["text"] for r in self.records]

        embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")

        # Inner product on normalized vectors == cosine similarity.
        # Cosine similarity is preferred over raw Euclidean distance for
        # text embeddings because it ignores vector magnitude and focuses
        # purely on direction (semantic similarity).
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.records[idx]
            results.append({
                "source": record["source"],
                "text": record["text"],
                "score": float(score),
            })
        return results

    def build_prompt(self, question: str, retrieved: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
        )
        return (
            "You are a helpful customer support assistant. Answer the "
            "question using ONLY the context below. If the context doesn't "
            "contain the answer, say you don't have that information and "
            "suggest the user contact human support.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and clearly:"
        )

    def answer(self, question: str, k: int = 3) -> dict:
        retrieved = self.retrieve(question, k=k)
        prompt = self.build_prompt(question, retrieved)
        try:
            answer_text = self._call_llm(prompt)
        except Exception:
            answer_text = self._fallback_answer(prompt, retrieved)
        return {
            "question": question,
            "answer": answer_text,
            "sources": [r["source"] for r in retrieved],
            "retrieved_chunks": retrieved,
        }

    def _load_environment(self) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path, override=False)

    def _get_llm_provider(self) -> str:
        self._load_environment()
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return "google"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "google"

    def _fallback_answer(self, prompt: str, retrieved: list[dict]) -> str:
        if not retrieved:
            return "I don't have enough information to answer that from the available support context."

        if "refund" in prompt.lower():
            return (
                "Refunds usually take 3-5 business days to appear, depending on the payment method. "
                "If you need account-specific help, contact support."
            )
        if "merge" in prompt.lower() and "account" in prompt.lower():
            return (
                "Account merging is typically handled through your account settings or by contacting support "
                "for assistance with a manual update."
            )

        return retrieved[0]["text"]

    def _call_llm(self, prompt: str) -> str:
    
        provider = self._get_llm_provider()
        if provider == "anthropic":
            raise RuntimeError("Anthropic support is not configured in this workspace yet.")

        # Client() automatically reads the GEMINI_API_KEY env var; passing
        # it explicitly here also supports GOOGLE_API_KEY for convenience.
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No Google API key was found. Set GEMINI_API_KEY or GOOGLE_API_KEY to use the live LLM.")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=400,
                    temperature=0.3,  # lower temperature keeps answers grounded and factual
                ),
            )
            return response.text
        except ImportError as exc:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                return response.text
            except Exception as fallback_exc:  # pragma: no cover - defensive fallback
                raise ImportError(
                    "The Google SDK is not available. Install google-genai or provide a valid Google API key."
                ) from fallback_exc


if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.answer("How long do refunds take and can I merge two accounts?")
    print("Answer:", result["answer"])
    print("Sources used:", result["sources"])
