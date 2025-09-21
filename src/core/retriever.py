import logging
import yaml
from typing import Optional

from langchain_core.retrievers import BaseRetriever

from src.core.vector_store import load_vector_store

logger = logging.getLogger(__name__)


def _load_retriever_cfg() -> dict:
    try:
        with open("configs/retriever.yml") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    cfg.setdefault("search_type", "similarity")
    cfg.setdefault("k", 4)
    return cfg


def get_retriever(k: Optional[int] = None) -> BaseRetriever:
    cfg = _load_retriever_cfg()
    search_type = cfg.get("search_type", "similarity")
    top_k = int(k) if k is not None else int(cfg.get("k", 4))

    db = load_vector_store()
    logger.info(f"Creating retriever: type={search_type}, k={top_k}")
    return db.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
