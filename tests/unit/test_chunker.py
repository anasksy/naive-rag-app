from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.core.chunker import chunk_documents

# Default config for mocking
DEFAULT_CFG = {
    "chunk_size": 50,
    "chunk_overlap": 10,
    "separators": ["\n\n", "\n", " ", ""],
}


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_basic_chunking(mock_cfg):
    """A document is correctly split into multiple chunks with overlap."""
    text = "A" * 120  # 120 characters -> 3 chunks (50, 50, 20)
    docs = [Document(page_content=text, metadata={"source": "test.txt"})]

    chunks = chunk_documents(docs)

    assert len(chunks) == 3
    # Check lengths
    assert all(len(c.page_content) <= 50 for c in chunks)
    # Check overlap: end of chunk 0 == beginning of chunk 1 (10 characters)
    tail = chunks[0].page_content[-10:]
    assert chunks[1].page_content.startswith(tail)
    # Check metadata is present
    assert chunks[0].metadata["source"] == "test.txt"
    assert "chunk" in chunks[0].metadata
    assert "source_doc" in chunks[0].metadata


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_multiple_documents(mock_cfg):
    """Multiple input documents are processed into a flat list of chunks."""
    d1 = Document(page_content="X" * 60, metadata={"source": "doc1"})
    d2 = Document(page_content="Y" * 60, metadata={"source": "doc2"})

    chunks = chunk_documents([d1, d2])

    # Both docs are chunked
    assert len(chunks) >= 2
    # source_doc differs between documents
    assert {c.metadata["source_doc"] for c in chunks} == {0, 1}


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_empty_document_returns_no_chunks(mock_cfg):
    """Empty content -> no chunks (current behavior of the splitter)."""
    docs = [Document(page_content="", metadata={"source": "empty"})]
    chunks = chunk_documents(docs)
    assert chunks == []


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_hard_cut_without_separators(mock_cfg):
    """No separator in text -> hard cut at chunk_size."""
    text = "A" * 130  # No whitespaces
    docs = [Document(page_content=text, metadata={})]

    chunks = chunk_documents(docs)
    assert [len(c.page_content) for c in chunks] == [50, 50, 50]


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_metadata_propagation(mock_cfg):
    """Original metadata is preserved and extended."""
    meta = {"source": "doc.pdf", "page": 3}
    d = Document(page_content="Hello World " * 10, metadata=meta)

    chunks = chunk_documents([d])
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["source"] == "doc.pdf"
        assert c.metadata["page"] == 3
        assert "chunk" in c.metadata
        assert "source_doc" in c.metadata


def test_bad_config_raises_keyerror():
    """Missing keys in the config currently lead to KeyError."""
    bad_cfg = {"chunk_overlap": 10}  # chunk_size is missing
    with patch("src.core.chunker.yaml.safe_load", return_value=bad_cfg):
        d = Document(page_content="abc", metadata={})
        with pytest.raises(KeyError):
            chunk_documents([d])


@patch("src.core.chunker.yaml.safe_load", return_value=DEFAULT_CFG)
def test_logging_messages(mock_cfg, caplog):
    """Important log messages appear."""
    d = Document(page_content="A" * 60, metadata={})
    with caplog.at_level("INFO"):
        chunk_documents([d])
    assert "Starting to chunk" in caplog.text
    assert "Chunking completed" in caplog.text
