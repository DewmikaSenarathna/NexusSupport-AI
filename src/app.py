"""
app.py

ENDPOINTS:
  GET  /health              — liveness check for load balancers/monitoring
  POST /classify             — classical ML classification only (fast, cheap)
  POST /ask                  — RAG: retrieve + LLM-generated answer
  POST /agent                — full agent: classify -> retrieve -> answer/escalate
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .classical_ml import load_classifier, classify_ticket
    from .rag_pipeline import RAGPipeline
    from .agent import SupportAgent
except ImportError:  # pragma: no cover - supports script execution
    from classical_ml import load_classifier, classify_ticket
    from rag_pipeline import RAGPipeline
    from agent import SupportAgent

ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_state["classifier"] = load_classifier("models/ticket_classifier.joblib")
    ml_state["rag"] = RAGPipeline(kb_dir="data/knowledge_base")
    ml_state["agent"] = SupportAgent(
        classifier_path="models/ticket_classifier.joblib",
        kb_dir="data/knowledge_base",
    )
    ml_state["request_log"] = []  # powers the dashboard
    yield
    ml_state.clear()


app = FastAPI(
    title="NexusSupport AI API",
    description="Classification, RAG, and agent endpoints for support ticket handling.",
    version="1.0.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The customer message or ticket text")


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class AgentResponse(BaseModel):
    response: str
    category: str
    escalated: bool
    sources: list[str] | None = None
    trace: list


def _log_request(endpoint: str, latency_ms: float, category: str | None = None):
    ml_state["request_log"].append({
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "category": category,
        "timestamp": time.time(),
    })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: TicketRequest):
    start = time.time()
    try:
        result = classify_ticket(ml_state["classifier"], req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    _log_request("/classify", (time.time() - start) * 1000, result["category"])
    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: TicketRequest):
    start = time.time()
    try:
        result = ml_state["rag"].answer(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    _log_request("/ask", (time.time() - start) * 1000)
    return {"question": result["question"], "answer": result["answer"], "sources": result["sources"]}


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: TicketRequest):
    start = time.time()
    try:
        result = ml_state["agent"].handle(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    _log_request("/agent", (time.time() - start) * 1000, result.get("category"))
    return result


@app.get("/metrics")
def metrics():
    """Powers the Streamlit dashboard: request volume, latency, category mix."""
    return {"requests": ml_state.get("request_log", [])}
