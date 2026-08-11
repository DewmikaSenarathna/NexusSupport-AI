"""
agent.py
"""

import os
from dataclasses import dataclass, field

try:
    from .classical_ml import load_classifier, classify_ticket
    from .rag_pipeline import RAGPipeline
except ImportError:  # pragma: no cover - supports script execution
    from classical_ml import load_classifier, classify_ticket
    from rag_pipeline import RAGPipeline

CONFIDENCE_ESCALATION_THRESHOLD = 0.35


@dataclass
class AgentTrace:
    steps: list = field(default_factory=list)

    def log(self, step: str, detail: dict):
        self.steps.append({"step": step, "detail": detail})


class SupportAgent:
    def __init__(self, classifier_path: str = "models/ticket_classifier.joblib",
                 kb_dir: str = "data/knowledge_base"):
        self.classifier = load_classifier(classifier_path)
        self.rag = RAGPipeline(kb_dir=kb_dir)

    def handle(self, message: str) -> dict:
        trace = AgentTrace()

        # --- Tool 1: classify ---
        classification = classify_ticket(self.classifier, message)
        trace.log("classify_ticket", classification)

        # --- Decision point: escalate if the model is unsure ---
        if classification["confidence"] < CONFIDENCE_ESCALATION_THRESHOLD:
            trace.log("decision", {"action": "escalate", "reason": "low classification confidence"})
            return {
                "response": (
                    "I'm not fully sure how to categorize this — I'm routing "
                    "it to a human support agent to make sure you get the "
                    "right help."
                ),
                "category": classification["category"],
                "escalated": True,
                "trace": trace.steps,
            }

        # --- Tool 2: retrieve relevant knowledge ---
        retrieved = self.rag.retrieve(message, k=3)
        trace.log("search_knowledge_base", {"num_chunks": len(retrieved),
                                             "sources": [r["source"] for r in retrieved]})

        # --- Decision point: escalate if retrieval found nothing relevant ---
        top_score = retrieved[0]["score"] if retrieved else 0.0
        if top_score < 0.2:
            trace.log("decision", {"action": "escalate", "reason": "no relevant knowledge found"})
            return {
                "response": (
                    "I couldn't find a confident answer in our knowledge "
                    "base for this. Escalating to a human agent."
                ),
                "category": classification["category"],
                "escalated": True,
                "trace": trace.steps,
            }

        # --- Tool 3: generate grounded answer ---
        prompt = self.rag.build_prompt(message, retrieved)
        answer_text = self.rag._call_llm(prompt)
        trace.log("answer_with_context", {"answer_preview": answer_text[:120]})

        return {
            "response": answer_text,
            "category": classification["category"],
            "urgency_hint": classification["all_scores"],
            "sources": [r["source"] for r in retrieved],
            "escalated": False,
            "trace": trace.steps,
        }


LANGCHAIN_REFERENCE_SKETCH = '''
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI

def classify_tool_fn(text: str) -> str:
    return str(classify_ticket(classifier, text))

def search_kb_tool_fn(query: str) -> str:
    results = rag.retrieve(query, k=3)
    return "\\n".join(r["text"] for r in results)

tools = [
    Tool(name="ClassifyTicket", func=classify_tool_fn,
         description="Classify a support ticket into a category."),
    Tool(name="SearchKnowledgeBase", func=search_kb_tool_fn,
         description="Search the FAQ knowledge base for relevant info."),
]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("The app crashes when I upload a file, what should I do?")
'''


if __name__ == "__main__":
    agent = SupportAgent()
    result = agent.handle("I was charged twice for my Pro plan, can I get a refund?")
    print("Response:", result["response"])
    print("Category:", result["category"])
    print("Trace:")
    for step in result["trace"]:
        print(" -", step)
