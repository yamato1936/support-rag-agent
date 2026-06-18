from typing import List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.retriever import SupportRetriever
from app.dense_retriever import DenseRetriever
from app.hybrid_retriever import HybridRetriever


app = FastAPI(title="support-rag-agent")


# Initialize retrievers once at startup.
# Dense and Hybrid load sentence-transformer models, so this may take a few seconds.
bm25_retriever = SupportRetriever()
dense_retriever = DenseRetriever()
hybrid_retrievers = {}


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    retriever: Literal["bm25", "dense", "hybrid"] = "bm25"
    alpha: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="BM25 weight for hybrid retrieval. Only used when retriever='hybrid'.",
    )


class RetrievedDocument(BaseModel):
    rank: int
    doc_id: str
    title: str
    score: float
    snippet: str
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_norm: Optional[float] = None
    dense_norm: Optional[float] = None
    alpha: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    retriever: str
    top_k: int
    alpha: Optional[float] = None
    retrieved_docs: List[RetrievedDocument]
    citations: List[str]


def get_hybrid_retriever(alpha: float) -> HybridRetriever:
    # Cache by alpha so repeated calls do not reload the dense model.
    key = round(alpha, 4)
    if key not in hybrid_retrievers:
        hybrid_retrievers[key] = HybridRetriever(alpha=key)
    return hybrid_retrievers[key]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.retriever == "bm25":
        docs = bm25_retriever.retrieve(req.question, top_k=req.top_k)
        response_alpha = None

    elif req.retriever == "dense":
        docs = dense_retriever.retrieve(req.question, top_k=req.top_k)
        response_alpha = None

    else:
        alpha = 0.3 if req.alpha is None else req.alpha
        retriever = get_hybrid_retriever(alpha)
        docs = retriever.retrieve(req.question, top_k=req.top_k)
        response_alpha = alpha

    return QueryResponse(
        question=req.question,
        retriever=req.retriever,
        top_k=req.top_k,
        alpha=response_alpha,
        retrieved_docs=docs,
        citations=[doc["doc_id"] for doc in docs],
    )
