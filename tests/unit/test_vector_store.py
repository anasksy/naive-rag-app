import os
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from src.core.vector_store import _load_vs_cfg, embed_and_store, load_vector_store


def test_load_vs_cfg_default_local():
    """Test loading default local vector store config"""
    with patch.dict(os.environ, {}, clear=True):
        with patch("builtins.open", MagicMock()), patch("yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "default": "local",
                "local": {
                    "provider": "chroma_local",
                    "persist_dir": "data/chroma",
                    "collection_name": "default"
                }
            }
            cfg = _load_vs_cfg()
            assert cfg["provider"] == "chroma_local"
            assert cfg["persist_dir"] == "data/chroma"
            assert cfg["collection_name"] == "default"


def test_load_vs_cfg_with_env_profile():
    """Test loading vector store config with environment profile override"""
    with patch.dict(os.environ, {"VS_PROFILE": "cloud"}):
        with patch("builtins.open", MagicMock()), patch("yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "default": "local",
                "cloud": {
                    "provider": "chroma_cloud",
                    "api_key_env": "CHROMA_API_KEY",
                    "tenant": "test-tenant",
                    "database": "test-db",
                    "collection_name": "default"
                }
            }
            cfg = _load_vs_cfg()
            assert cfg["provider"] == "chroma_cloud"
            assert cfg["collection_name"] == "default"


def test_load_vs_cfg_missing_profile():
    """Test loading vector store config with missing profile"""
    with patch.dict(os.environ, {"VS_PROFILE": "nonexistent"}):
        with patch("builtins.open", MagicMock()), patch("yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "default": "local",
                "local": {
                    "provider": "chroma_local",
                    "persist_dir": "data/chroma",
                    "collection_name": "default"
                }
            }
            
            with pytest.raises(KeyError, match="profile 'nonexistent' not found"):
                _load_vs_cfg()


@patch("src.core.vector_store.Chroma")
@patch("src.core.vector_store.get_embedder")
def test_embed_and_store_chroma_local(mock_get_embedder, mock_chroma):
    """Test embedding and storing documents with Chroma local"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock the Chroma instance
    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db
    
    # Mock config
    mock_cfg = {
        "provider": "chroma_local",
        "persist_dir": "test/chroma",
        "collection_name": "test_collection"
    }
    
    # Create test documents
    docs = [
        Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test content 2", metadata={"source": "test2.txt"})
    ]
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        result = embed_and_store(docs)
        
        # Verify the Chroma.from_documents was called with correct parameters
        mock_chroma.from_documents.assert_called_once_with(
            documents=docs,
            embedding=mock_embedder,
            persist_directory="test/chroma",
            collection_name="test_collection"
        )
        
        # Verify the result is the mock db
        assert result == mock_db


@patch("src.core.vector_store.Chroma")
@patch("src.core.vector_store.get_embedder")
def test_embed_and_store_chroma_cloud(mock_get_embedder, mock_chroma):
    """Test embedding and storing documents with Chroma cloud"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock the Chroma instance
    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db
    
    # Mock config
    mock_cfg = {
        "provider": "chroma_cloud",
        "collection_name": "test_collection"
    }
    
    # Create test documents
    docs = [
        Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test content 2", metadata={"source": "test2.txt"})
    ]
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        result = embed_and_store(docs)
        
        # Verify the Chroma.from_documents was called with correct parameters (no persist_directory)
        mock_chroma.from_documents.assert_called_once_with(
            documents=docs,
            embedding=mock_embedder,
            collection_name="test_collection"
        )
        
        # Verify the result is the mock db
        assert result == mock_db


@patch("src.core.vector_store.get_embedder")
def test_embed_and_store_unknown_provider(mock_get_embedder):
    """Test embedding and storing with unknown provider"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock config with unknown provider
    mock_cfg = {
        "provider": "unknown_provider",
        "collection_name": "test_collection"
    }
    
    # Create test documents
    docs = [Document(page_content="Test content", metadata={"source": "test.txt"})]
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            embed_and_store(docs)


@patch("src.core.vector_store.Chroma")
@patch("src.core.vector_store.get_embedder")
def test_load_vector_store_chroma_local(mock_get_embedder, mock_chroma):
    """Test loading existing Chroma local store"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock the Chroma instance
    mock_db = MagicMock()
    mock_chroma.return_value = mock_db
    
    # Mock config
    mock_cfg = {
        "provider": "chroma_local",
        "persist_dir": "test/chroma",
        "collection_name": "test_collection"
    }
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        result = load_vector_store()
        
        # Verify the Chroma was called with correct parameters
        mock_chroma.assert_called_once_with(
            persist_directory="test/chroma",
            collection_name="test_collection",
            embedding_function=mock_embedder
        )
        
        # Verify the result is the mock db
        assert result == mock_db


@patch("src.core.vector_store.Chroma")
@patch("src.core.vector_store.get_embedder")
def test_load_vector_store_chroma_cloud(mock_get_embedder, mock_chroma):
    """Test loading existing Chroma cloud store"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock the Chroma instance
    mock_db = MagicMock()
    mock_chroma.return_value = mock_db
    
    # Mock config
    mock_cfg = {
        "provider": "chroma_cloud",
        "collection_name": "test_collection"
    }
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        result = load_vector_store()
        
        # Verify the Chroma was called with correct parameters (no persist_directory)
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embedder
        )
        
        # Verify the result is the mock db
        assert result == mock_db


@patch("src.core.vector_store.get_embedder")
def test_load_vector_store_unknown_provider(mock_get_embedder):
    """Test loading vector store with unknown provider"""
    # Mock the embedder
    mock_embedder = MagicMock()
    mock_get_embedder.return_value = mock_embedder
    
    # Mock config with unknown provider
    mock_cfg = {
        "provider": "unknown_provider",
        "collection_name": "test_collection"
    }
    
    with patch("src.core.vector_store._load_vs_cfg", return_value=mock_cfg):
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            load_vector_store()