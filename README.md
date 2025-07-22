# Naive RAG Application

This application implements a modular Retrieval-Augmented Generation (RAG) pipeline — covering the essential steps required to transform documents into indexed chunks and generate answers with context and source citations.

---

## 🎯 Features

1. **Ingest documents** (`PDF`, `TXT`, `MD`, `DOCX`, `CSV`)  
2. **Split into chunks** using configurable strategies with LangChain  
3. **Embed & index** with OpenAI, HuggingFace, or Anthropic embeddings into ChromaDB  
4. **Retrieve context** via vector similarity with advanced filtering  
5. **Answer generation** via multiple LLM providers (OpenAI, HuggingFace, Anthropic)  
6. **Query via UI** (Streamlit) or API (FastAPI)
7. **Comprehensive logging** for debugging and monitoring
8. **CI/CD ready** with GitHub Actions for testing, linting, and type checking

---

## 📦 Tech Stack

| Layer                | Tech / Tool                    |
| -------------------- | ------------------------------ |
| **Backend**          | Python, FastAPI, Uvicorn       |
| **Orchestration**    | LangChain (v0.3+)              |
| **Vector Store**     | ChromaDB                       |
| **LLM Providers**    | OpenAI, HuggingFace, Anthropic |
| **Embedding**        | OpenAI, HuggingFace            |
| **Frontend**         | Streamlit                      |
| **Deps & Packaging** | Poetry                         |
| **Lint & Format**    | Ruff                           |
| **Testing**          | pytest, pytest-cov             |
| **Type Checking**    | mypy                           |
| **Env Management**   | python-dotenv                  |
| **CI/CD**            | GitHub Actions                 |

---

## 🚀 Getting Started

```bash
# 1. Clone & enter
git clone <repo-url>
cd <repo>

# 2. Install Poetry (once globally)
pip install poetry

# 3. Create venv & install deps
poetry install

# 4. Copy env template & add keys
cp .env.example .env
# → Fill in OPENAI_API_KEY, HF_API_KEY, or other LLM provider credentials

# 5. Run linting, tests, type-check
poetry run ruff check . --fix
poetry run pytest
poetry run mypy src

# 6. Launch Streamlit UI
poetry run streamlit run src/ui/app.py
