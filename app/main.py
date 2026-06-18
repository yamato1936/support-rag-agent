from typing import List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.retriever import SupportRetriever
from app.dense_retriever import DenseRetriever
from app.hybrid_retriever import HybridRetriever
from app.generator import GroundedAnswerGenerator


app = FastAPI(title="support-rag-agent")


# Initialize retrievers once at startup.
bm25_retriever = SupportRetriever()
dense_retriever = DenseRetriever()
hybrid_retrievers = {}

answer_generator = GroundedAnswerGenerator()


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


class AnswerRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    alpha: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    retriever: str
    top_k: int
    alpha: Optional[float] = None
    citations: List[str]
    is_supported: bool
    retrieved_docs: List[RetrievedDocument]


def get_hybrid_retriever(alpha: float) -> HybridRetriever:
    key = round(alpha, 4)
    if key not in hybrid_retrievers:
        hybrid_retrievers[key] = HybridRetriever(alpha=key)
    return hybrid_retrievers[key]


def retrieve_docs(
    question: str,
    top_k: int,
    retriever_name: Literal["bm25", "dense", "hybrid"],
    alpha: Optional[float],
):
    if retriever_name == "bm25":
        docs = bm25_retriever.retrieve(question, top_k=top_k)
        response_alpha = None

    elif retriever_name == "dense":
        docs = dense_retriever.retrieve(question, top_k=top_k)
        response_alpha = None

    else:
        hybrid_alpha = 0.3 if alpha is None else alpha
        retriever = get_hybrid_retriever(hybrid_alpha)
        docs = retriever.retrieve(question, top_k=top_k)
        response_alpha = hybrid_alpha

    return docs, response_alpha


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    docs, response_alpha = retrieve_docs(
        question=req.question,
        top_k=req.top_k,
        retriever_name=req.retriever,
        alpha=req.alpha,
    )

    return QueryResponse(
        question=req.question,
        retriever=req.retriever,
        top_k=req.top_k,
        alpha=response_alpha,
        retrieved_docs=docs,
        citations=[doc["doc_id"] for doc in docs],
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    docs, response_alpha = retrieve_docs(
        question=req.question,
        top_k=req.top_k,
        retriever_name=req.retriever,
        alpha=req.alpha,
    )

    generated = answer_generator.generate(
        question=req.question,
        retrieved_docs=docs,
    )

    return AnswerResponse(
        question=req.question,
        answer=generated["answer"],
        retriever=req.retriever,
        top_k=req.top_k,
        alpha=response_alpha,
        citations=generated["citations"],
        is_supported=generated["is_supported"],
        retrieved_docs=docs,
    )
