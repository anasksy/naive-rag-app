import os
import yaml
import logging

from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

logger = logging.getLogger(__name__)


def _load_embed_cfg() -> dict:
    with open("configs/embeddings.yml") as f:
        cfg = yaml.safe_load(f)
    profile = os.getenv("EMBED_PROFILE", cfg.get("default", "local"))
    return cfg[profile]


def get_embedder() -> Embeddings:
    cfg = _load_embed_cfg()
    provider = cfg.get("provider", "ollama")

    if provider == "ollama":
        model = cfg.get("model_name", "mxbai-embed-large")
        base_url = cfg.get("base_url", "http://localhost:11434")
        logger.info(f"Using OllamaEmbeddings: {model} ({base_url})")
        return OllamaEmbeddings(model=model, base_url=base_url)

    if provider == "huggingface_hub":       
        model = cfg.get("model_name", "mxbai-embed-large-v1")
        token = os.getenv(cfg.get("api_key_env", "HF_TOKEN"))
        if not token:
            raise EnvironmentError("HF token missing. Set HF_TOKEN env var.")
        logger.info(f"Using HuggingFaceEndopointEmbeddings: {model}")
        return HuggingFaceEndpointEmbeddings(
            repo_id=model,
            huggingfacehub_api_token=token
        )

    raise ValueError(f"Unknown embeddings provider: {provider}")