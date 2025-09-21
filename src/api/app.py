from __future__ import annotations

import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.rag import answer

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Naive RAG API",
    description=("Very small Retrieval Augmented Generation API"),
)


class SourceItem(BaseModel):
    source: str = Field(..., description="Original document identifier")
    chunk: Optional[int] = Field(None, description="Chunk index inside the document")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight readiness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline for a user question."""
    logger.info(
        "Received query", extra={"question": payload.question, "top_k": payload.top_k}
    )
    try:
        rag_result = answer(payload.question, k=payload.top_k)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [SourceItem(**s) for s in rag_result.get("sources", [])]
    return QueryResponse(answer=rag_result.get("answer", ""), sources=sources)


__all__ = [
    "app",
    "QueryRequest",
    "QueryResponse",
    "SourceItem",
]
