import logging
import os
from typing import Any, Optional, cast

import yaml
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI

try:  # pragma: no cover - optional dependency
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    _ChatGoogleGenerativeAI = None

ChatGoogleGenerativeAI = cast(Any, _ChatGoogleGenerativeAI)

logger = logging.getLogger(__name__)


def _load_llm_cfg() -> dict:
    """Load LLM profile from YAML with safe defaults."""
    try:
        with open("configs/llm.yml") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    default_profile = cfg.get("default", "openai")
    profile = os.getenv("LLM_PROFILE", default_profile)
    section = cfg.get(profile, {})

    # Provide minimal defaults for OpenAI
    if profile == "openai":
        section.setdefault("provider", "openai")
        section.setdefault("model_name", "gpt-5-nano")
    elif profile == "huggingface":
        section.setdefault("provider", "huggingface")
        section.setdefault("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    else:
        # If custom profile, ensure provider key exists
        section.setdefault("provider", "openai")

    return section


def get_llm(model: Optional[str] = None):
    """Return a simple chat LLM based on config/env.

    Supports providers:
    - openai: requires OPENAI_API_KEY
    - huggingface: requires HF_TOKEN
    """
    cfg = _load_llm_cfg()
    provider = cfg.get("provider", "openai")
    # Priority: explicit arg > config
    candidate_model = model or cfg.get("model_name")
    if not isinstance(candidate_model, str) or not candidate_model:
        raise ValueError("LLM model name must be a non-empty string in configuration")
    model_name = candidate_model

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY missing. Set it in your environment."
            )
        logger.info(f"Using OpenAI Chat model: {model_name}")
        # ChatOpenAI reads api key from env; we pass model only
        return ChatOpenAI(model=model_name)

    if provider == "huggingface":
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN missing. Set it in your environment.")
        logger.info(f"Using HuggingFace Chat model: {model_name}")
        return ChatHuggingFace(repo_id=model_name, token=token)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY missing. Set it in your environment."
            )
        logger.info(f"Using Gemini Chat model: {model_name}")
        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "langchain-google-genai not installed. Add it to dependencies."
            )
        # ChatGoogleGenerativeAI reads GOOGLE_API_KEY from env
        return ChatGoogleGenerativeAI(model=model_name)

    raise ValueError(f"Unknown LLM provider: {provider}")
