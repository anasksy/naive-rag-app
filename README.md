# Naive Rag Application

This application implements the minimal version of a Retrieval-Augmented Generation (RAG) pipeline — covering only the essential steps required to transform documents into indexed chunks and generate answers with context and source citations

---

## 🎯 Features

1. **Ingest documents** (`PDF`, `TXT`, `MD`, `DOCX`)  
2. **Chunk text** into overlapping passages  
3. **Embed & index** with OpenAI embeddings + ChromaDB  
4. **Retrieve & answer** with LLMs via LangChain
5. **End-to-end UI** for querying and source display

---

## 📦 Tech Stack

| Layer                | Tech / Tool              |
| -------------------- | ------------------------ |
| **Backend**          | Python, FastAPI, Uvicorn |
| **Orchestration**    | LangChain                |
| **Vector Store**     | ChromaDB                 |
| **LLM**              | Any                      |
| **Frontend**         | Streamlit                |
| **Deps & Packaging** | Poetry                   |
| **Lint & Format**    | Ruff                     |
| **Testing**          | pytest                   |
| **Type Checking**    | mypy                     |
| **Env Management**   | python-dotenv            |

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
