from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("src.api.app.answer")
def test_query_endpoint_returns_rag_output(mock_answer):
    mock_answer.return_value = {
        "answer": "42",
        "sources": [{"source": "doc.txt", "chunk": 0}],
    }

    resp = client.post(
        "/query",
        json={"question": "What is the answer?", "top_k": 2},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "answer": "42",
        "sources": [{"source": "doc.txt", "chunk": 0}],
    }
    mock_answer.assert_called_once_with("What is the answer?", k=2)


def test_query_validation_for_missing_question():
    resp = client.post("/query", json={})
    assert resp.status_code == 422
