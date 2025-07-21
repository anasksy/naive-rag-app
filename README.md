# Naive Rag Application

This application implements the minimal version of a Retrieval-Augmented Generation (RAG) pipeline — covering only the essential steps required to transform documents into indexed chunks and generate answers with context and source citations

---

## 🎯 Features

1. **Ingest documents** (`PDF`, `TXT`, `MD`, `DOCX`)  
2. **Split into chunks** using a tokenizer (configurable)  
3. **Embed & index** with HuggingFace or OpenAI embeddings into ChromaDB  
4. **Retrieve context** via vector similarity  
5. **Answer generation** via LLM (OpenAI, Groq, HuggingFace)  
6. **Query via UI** (Streamlit) or API (FastAPI)
7. **CI/CD ready** (with linting, testing, typing via GitHub Actions)

---

## 📦 Tech Stack

| Layer                | Tech / Tool              |
| -------------------- | ------------------------ |
| **Backend**          | Python, FastAPI, Uvicorn |
| **Orchestration**    | LangChain                |
| **Vector Store**     | ChromaDB                 |
| **LLM**              | Groq (Llama 3), OpenAI                      |
**Embedding**|  HuggingFace, OpenAI
| **Frontend**         | Streamlit                |
| **Deps & Packaging** | Poetry                   |
| **Lint & Format**    | Ruff                     |
| **Testing**          | pytest                   |
| **Type Checking**    | mypy                     |
| **Env Management**   | python-dotenv            |
| **CI/CD**   | GitHub Actions
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
# → Fill in OPENAI_API_KEY, HF_API_KEY, or GROQ credentials

# 5. Run linting, tests, type-check
poetry run ruff check . --fix
poetry run pytest
poetry run mypy src

# 6. Launch Streamlit UI
poetry run streamlit run src/ui/app.py
