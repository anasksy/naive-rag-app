from langchain_core.documents import Document

from src.core.chunker import chunk_documents
from src.core.loader import load_documents


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
