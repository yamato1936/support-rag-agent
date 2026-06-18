from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from app.retriever import SupportRetriever


app = FastAPI(title="support-rag-agent")

retriever = SupportRetriever()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RetrievedDocument(BaseModel):
    rank: int
    doc_id: str
    title: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    retrieved_docs: List[RetrievedDocument]
    citations: List[str]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    docs = retriever.retrieve(req.question, top_k=req.top_k)

    return QueryResponse(
        question=req.question,
        retrieved_docs=docs,
        citations=[doc["doc_id"] for doc in docs],
    )