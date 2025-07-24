import os
from unittest.mock import patch, MagicMock
import pytest

from src.core.embedder import _load_embed_cfg, get_embedder


def test_load_embed_cfg_default_local():
    """Test loading default local config"""
    with patch.dict(os.environ, {}, clear=True):
        cfg = _load_embed_cfg()
        assert cfg["provider"] == "ollama"
        assert cfg["model_name"] == "mxbai-embed-large"
        assert cfg["base_url"] == "http://localhost:11434"


def test_load_embed_cfg_with_env_profile():
    """Test loading config with environment profile override"""
    with patch.dict(os.environ, {"EMBED_PROFILE": "cloud"}):
        cfg = _load_embed_cfg()
        assert cfg["provider"] == "huggingface_hub"
        assert cfg["model_name"] == "mxbai-embed-large-v1"


@patch("src.core.embedder.OllamaEmbeddings")
def test_get_embedder_ollama(mock_ollama):
    """Test getting Ollama embedder"""
    mock_instance = MagicMock()
    mock_ollama.return_value = mock_instance
    
    with patch("src.core.embedder._load_embed_cfg") as mock_load_cfg:
        mock_load_cfg.return_value = {
            "provider": "ollama",
            "model_name": "test-model",
            "base_url": "http://test:11434"
        }
        
        embedder = get_embedder()
        
        mock_ollama.assert_called_once_with(model="test-model", base_url="http://test:11434")
        assert embedder == mock_instance


@patch("src.core.embedder.HuggingFaceEndpointEmbeddings")
def test_get_embedder_huggingface_hub(mock_hf):
    """Test getting HuggingFace Hub embedder"""
    mock_instance = MagicMock()
    mock_hf.return_value = mock_instance
    
    with patch("src.core.embedder._load_embed_cfg") as mock_load_cfg:
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            mock_load_cfg.return_value = {
                "provider": "huggingface_hub",
                "model_name": "test-model",
                "api_key_env": "HF_TOKEN"
            }
            
            embedder = get_embedder()
            
            mock_hf.assert_called_once_with(
                repo_id="test-model",
                huggingfacehub_api_token="test-token"
            )
            assert embedder == mock_instance


def test_get_embedder_huggingface_hub_missing_token():
    """Test HuggingFace Hub embedder with missing token"""
    with patch("src.core.embedder._load_embed_cfg") as mock_load_cfg:
        with patch.dict(os.environ, {}, clear=True):
            mock_load_cfg.return_value = {
                "provider": "huggingface_hub",
                "model_name": "test-model",
                "api_key_env": "HF_TOKEN"
            }
            
            with pytest.raises(EnvironmentError, match="HF token missing"):
                get_embedder()


def test_get_embedder_unknown_provider():
    """Test getting embedder with unknown provider"""
    with patch("src.core.embedder._load_embed_cfg") as mock_load_cfg:
        mock_load_cfg.return_value = {
            "provider": "unknown_provider"
        }
        
        with pytest.raises(ValueError, match="Unknown embeddings provider"):
            get_embedder()


@patch("src.core.embedder.OllamaEmbeddings")
def test_get_embedder_ollama_default_values(mock_ollama):
    """Test getting Ollama embedder with default values"""
    mock_instance = MagicMock()
    mock_ollama.return_value = mock_instance
    
    with patch("src.core.embedder._load_embed_cfg") as mock_load_cfg:
        mock_load_cfg.return_value = {
            "provider": "ollama"
            # No model_name or base_url specified, should use defaults
        }
        
        embedder = get_embedder()
        
        mock_ollama.assert_called_once_with(
            model="mxbai-embed-large",
            base_url="http://localhost:11434"
        )
        assert embedder == mock_instance