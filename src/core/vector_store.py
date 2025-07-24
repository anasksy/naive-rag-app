import logging
import os
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .embedder import get_embedder

logger = logging.getLogger(__name__)


def _load_vs_cfg() -> dict:
    """Load vector store profile from YAML, fallback to 'local'."""
    with open("configs/vector_store.yml") as f:
        cfg = yaml.safe_load(f) or {}

    default_profile = cfg.get("default", "local")
    profile = os.getenv("VS_PROFILE", default_profile)
    section = cfg.get(profile)
    if section is None:
        raise KeyError(
            f"profile '{profile}' not found in vector_store.yml. "
            f"Available keys: {list(cfg.keys())}"
        )
    section.setdefault("provider", "chroma_local")
    return section


def embed_and_store(docs: List[Document]) -> Chroma:
    """
    Embed chunks and store them (local persist or cloud) with LangChain's Chroma wrapper.
    Returns the Chroma instance for immediate querying.
    """
    vcfg = _load_vs_cfg()
    provider = vcfg.get("provider", "chroma_local")
    collection = vcfg.get("collection_name", "default")

    emb = get_embedder()

    if provider == "chroma_local":
        persist_dir = vcfg["persist_dir"]
        logger.info(f"Embedding {len(docs)} docs -> {persist_dir} ({collection})")
        db = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            persist_directory=persist_dir,
            collection_name=collection,
        )
    elif provider == "chroma_cloud":
        logger.info(f"Embedding {len(docs)} docs -> chroma_cloud ({collection})")
        # Kein persist_directory bei Cloud
        db = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            collection_name=collection,
        )
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

    logger.info("Embeddings stored.")
    return db


def load_vector_store() -> Chroma:
    """
    Re-open an existing Chroma store (local or cloud) using the same embedder.
    """
    vcfg = _load_vs_cfg()
    provider = vcfg.get("provider", "chroma_local")
    collection = vcfg.get("collection_name", "default")

    emb = get_embedder()

    if provider == "chroma_local":
        persist_dir = vcfg["persist_dir"]
        logger.info(f"Loading Chroma @ {persist_dir} ({collection})")
        return Chroma(
            persist_directory=persist_dir,
            collection_name=collection,
            embedding_function=emb,
        )
    elif provider == "chroma_cloud":
        logger.info(f"Loading Chroma Cloud ({collection})")
        return Chroma(
            collection_name=collection,
            embedding_function=emb,
        )
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
