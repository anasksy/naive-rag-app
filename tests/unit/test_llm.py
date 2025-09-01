from unittest.mock import patch, MagicMock
import os
import pytest

from src.core.llm import _load_llm_cfg, get_llm


def test_load_llm_cfg_defaults():
    # Patch file IO and yaml parsing; we don't need the open-handle itself
    with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {
            "default": "openai",
            "openai": {"provider": "openai", "model_name": "gpt-5-nano"},
        }
        cfg = _load_llm_cfg()
        assert cfg["provider"] == "openai"
        assert cfg["model_name"] == "gpt-5-nano"


@patch("src.core.llm.ChatOpenAI")
def test_get_llm_openai(mock_openai):
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance

    with patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True):
        with patch("src.core.llm._load_llm_cfg") as mock_cfg:
            mock_cfg.return_value = {"provider": "openai", "model_name": "gpt-5-nano"}
            llm = get_llm()
            mock_openai.assert_called_once_with(model="gpt-5-nano")
            assert llm == mock_instance


def test_get_llm_openai_missing_key():
    with patch("src.core.llm._load_llm_cfg") as mock_cfg:
        mock_cfg.return_value = {"provider": "openai", "model_name": "gpt-5-nano"}
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                get_llm()


@patch("src.core.llm.ChatHuggingFace")
def test_get_llm_huggingface(mock_hf):
    mock_instance = MagicMock()
    mock_hf.return_value = mock_instance

    with patch.dict(os.environ, {"HF_TOKEN": "y"}, clear=True):
        with patch("src.core.llm._load_llm_cfg") as mock_cfg:
            mock_cfg.return_value = {"provider": "huggingface", "model_name": "my-model"}
            llm = get_llm()
            mock_hf.assert_called_once_with(repo_id="my-model", token="y")
            assert llm == mock_instance


def test_get_llm_huggingface_missing_key():
    with patch("src.core.llm._load_llm_cfg") as mock_cfg:
        mock_cfg.return_value = {"provider": "huggingface", "model_name": "my-model"}
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="HF_TOKEN"):
                get_llm()


def test_get_llm_unknown_provider():
    with patch("src.core.llm._load_llm_cfg") as mock_cfg:
        mock_cfg.return_value = {"provider": "xyz", "model_name": "m"}
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()


@patch("src.core.llm.ChatGoogleGenerativeAI")
def test_get_llm_gemini(mock_gemini):
    mock_instance = MagicMock()
    mock_gemini.return_value = mock_instance

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "g"}, clear=True):
        with patch("src.core.llm._load_llm_cfg") as mock_cfg:
            mock_cfg.return_value = {"provider": "gemini", "model_name": "gemini-2.5-flash-lite"}
            llm = get_llm()
            mock_gemini.assert_called_once_with(model="gemini-2.5-flash-lite")
            assert llm == mock_instance
