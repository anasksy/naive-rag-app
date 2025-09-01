from unittest.mock import MagicMock, patch

from src.core.retriever import _load_retriever_cfg, get_retriever


def test_load_retriever_cfg_defaults(tmp_path, monkeypatch):
    # No file present -> defaults should be returned
    # ensure CWD has no configs/retriever.yml override
    cfg = _load_retriever_cfg()
    assert cfg["search_type"] == "similarity"
    assert cfg["k"] == 4


@patch("src.core.retriever.load_vector_store")
def test_get_retriever_uses_config(mock_load_vs):
    # Mock DB and its as_retriever
    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_load_vs.return_value = mock_db

    # Override config via patching the loader
    with patch("src.core.retriever._load_retriever_cfg") as mock_cfg:
        mock_cfg.return_value = {"search_type": "similarity", "k": 5}

        r = get_retriever()

        mock_db.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 5}
        )
        assert r == mock_retriever


@patch("src.core.retriever.load_vector_store")
def test_get_retriever_k_override(mock_load_vs):
    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_load_vs.return_value = mock_db

    with patch("src.core.retriever._load_retriever_cfg") as mock_cfg:
        mock_cfg.return_value = {"search_type": "similarity", "k": 3}

        r = get_retriever(k=7)

        mock_db.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 7}
        )
        assert r == mock_retriever

