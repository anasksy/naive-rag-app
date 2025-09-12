from unittest.mock import MagicMock, patch

from src.core.rag import answer, _build_prompt, _format_sources


def test_build_prompt_contains_context_and_question():
    p = _build_prompt("CTX", "What?")
    assert "CTX" in p and "What?" in p and "Answer" in p


def test_format_sources_from_docs():
    d1 = MagicMock(metadata={"source": "a.txt", "chunk": 1})
    d2 = MagicMock(metadata={"source": "b.txt", "chunk": 2})
    out = _format_sources([d1, d2])
    assert out == [
        {"source": "a.txt", "chunk": 1},
        {"source": "b.txt", "chunk": 2},
    ]

@patch("src.core.rag.get_retriever")
@patch("src.core.rag.get_llm")
def test_answer_happy_path(mock_get_llm, mock_get_retriever):
    # Mock retriever returning two docs
    doc1 = MagicMock(page_content="foo", metadata={"source": "a.txt", "chunk": 0})
    doc2 = MagicMock(page_content="bar", metadata={"source": "b.txt", "chunk": 1})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    mock_get_retriever.return_value = retriever

    # Mock LLM
    llm = MagicMock()
    msg = MagicMock(content="Answer text")
    llm.invoke.return_value = msg
    mock_get_llm.return_value = llm

    res = answer("question?", k=3)
    assert res["answer"] == "Answer text"
    assert res["sources"] == [
        {"source": "a.txt", "chunk": 0},
        {"source": "b.txt", "chunk": 1},
    ]
    mock_get_retriever.assert_called_once_with(k=3)
    retriever.invoke.assert_called_once()
    llm.invoke.assert_called_once()


@patch("src.core.rag.get_retriever")
@patch("src.core.rag.get_llm")
def test_answer_no_docs_returns_message(mock_get_llm, mock_get_retriever):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_get_retriever.return_value = retriever

    res = answer("question?")
    assert "No relevant information" in res["answer"]
    assert res["sources"] == []
    mock_get_llm.assert_not_called()
