import logging
from typing import Dict, List

from langchain_core.documents import Document

from src.core.retriever import get_retriever
from src.core.llm import get_llm

logger = logging.getLogger(__name__)


def _format_sources(docs: List[Document]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for d in docs:
        src = str(d.metadata.get("source", "unknown"))
        ch = d.metadata.get("chunk")
        sources.append({"source": src, "chunk": ch})
    return sources


def _build_prompt(context: str, question: str) -> str:
    return (
        "Answer factually using the provided context and elaborate with necessary detail. "
        "If the context does not contain the answer, explicitly say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer(query: str, k: int = 4) -> Dict[str, object]:
    """Minimal RAG call: Retriever -> Prompt -> LLM -> Answer + Sources.

    Returns a dict: {"answer": str, "sources": [{"source": str, "chunk": int}]}
    """
    retriever = get_retriever(k=k)
    logger.info(f"Retrieving top-{k} documents for query: {query!r}")
    # Prefer modern LangChain retriever API; fall back if needed
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:  # pragma: no cover - legacy path
        docs = retriever.get_relevant_documents(query)

    if not docs:
        return {"answer": "No relevant information found.", "sources": []}

    context = "\n\n".join(d.page_content for d in docs)
    prompt = _build_prompt(context, query)

    llm = get_llm()
    logger.info("Querying LLM with retrieved context")
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))

    return {"answer": text, "sources": _format_sources(docs)}
