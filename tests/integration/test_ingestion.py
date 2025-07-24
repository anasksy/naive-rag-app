import os
import tempfile
import shutil
from unittest.mock import patch
import pytest
from langchain_core.documents import Document

from src.core.chunker import chunk_documents
from src.core.loader import load_documents
from src.core.vector_store import embed_and_store, load_vector_store


def test_ingest_flow_loader_to_chunker(tmp_path):
    """Load file -> Documents -> chunk -> verify chunks."""
    # 1) Create test file
    content = "Intro\n\n" + "A " * 300 + "\n\nOutro"  # ~600+ chars
    fpath = tmp_path / "sample.txt"
    fpath.write_text(content)

    # 2) Load
    docs = load_documents(str(fpath))
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source"] == str(fpath)

    # 3) Chunk
    chunks = chunk_documents(docs)
    assert len(chunks) >= 2
    # Check order/metadata
    for i, ch in enumerate(chunks):
        assert "chunk" in ch.metadata
        assert "source_doc" in ch.metadata
        assert ch.metadata["source"] == str(fpath)
        # All chunks have text
        assert ch.page_content.strip() != ""

    # 4) Overlap sanity check (optional, only if at least 2 chunks)
    if len(chunks) > 1:
        # Check that chunks contain expected content parts
        all_content = "".join(chunk.page_content for chunk in chunks)
        assert "Intro" in all_content
        assert "A A A" in all_content  # Part of the repeated A's
        assert "Outro" in all_content


def test_ingest_flow_complete_pipeline(tmp_path):
    """Load file -> Documents -> chunk -> embed -> store -> verify retrieval."""
    # This test requires Ollama to be running locally
    if os.getenv("CI") == "true" or os.getenv("SKIP_OLLAMA_TESTS") == "true":
        pytest.skip("Skipping complete pipeline test in CI or when explicitly disabled")
    
    # Create a temporary directory for Chroma persistence
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1) Create test file
        content = "The quick brown fox jumps over the lazy dog. " * 20  # Repeated sentence
        fpath = tmp_path / "sample.txt"
        fpath.write_text(content)
        
        # Override the persist directory to use our temporary directory
        with patch.dict(os.environ, {
            "EMBED_PROFILE": "local",
            "VS_PROFILE": "local"
        }, clear=True):
            with patch("src.core.vector_store._load_vs_cfg") as mock_load_cfg:
                mock_load_cfg.return_value = {
                    "provider": "chroma_local",
                    "persist_dir": temp_dir,
                    "collection_name": "test_pipeline"
                }
                
                try:
                    # 2) Load
                    docs = load_documents(str(fpath))
                    assert len(docs) == 1
                    
                    # 3) Chunk
                    chunks = chunk_documents(docs)
                    assert len(chunks) >= 1
                    
                    # 4) Embed and store
                    db = embed_and_store(chunks)
                    
                    # 5) Verify storage worked
                    assert db is not None
                    
                    # 6) Load vector store
                    loaded_db = load_vector_store()
                    
                    # 7) Perform similarity search
                    query = "quick brown fox"
                    results = loaded_db.similarity_search(query, k=2)
                    
                    # 8) Verify we get relevant results
                    assert len(results) >= 1
                    assert all(hasattr(result, 'page_content') for result in results)
                    assert all(hasattr(result, 'metadata') for result in results)
                    
                    # 9) Verify that the content is relevant
                    assert any("quick brown fox" in result.page_content.lower() for result in results)
                    
                except Exception as e:
                    # If Ollama is not running or model is not available, skip the test
                    if "Connection error" in str(e) or "not found" in str(e):
                        pytest.skip(f"Ollama service or model not available: {e}")
                    else:
                        # Re-raise if it's a different error
                        raise
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
