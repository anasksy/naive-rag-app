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
| **Vector Store**     | ChromaDB (local & cloud)       |
| **LLM Providers**    | OpenAI, HuggingFace, Anthropic |
| **Embedding**        | Ollama, HuggingFace   |
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
```

---

## 🔄 Ingestion Pipeline

This project contains a complete ingestion pipeline for RAG applications:

1. **Document Loading**: Support for PDF, Text, Markdown and other formats
2. **Chunking**: Configurable text splitting with overlap
3. **Embedding**: Multiple provider options (Ollama, HuggingFace, OpenAI)
4. **Vector Storage**: Persistent storage for efficient retrieval

### Embeddings

The application supports the following embedding providers:

- **Ollama (local)**: Runs on your local system

  ```bash
  # Start Ollama before usage
  ollama serve
  # Pull the mxbai-embed-large model if not already available
  ollama pull mxbai-embed-large
  ```

- **HuggingFace (cloud)**: Uses the HuggingFace API

  ```bash
  # Set API token
  export HF_TOKEN="your-hf-api-token"
  # Choose HF as provider
  export EMBED_PROFILE="cloud"
  ```

### Vector Store

The application supports the following vector store options:

- **Chroma (local)**: Persistent local storage

  ```bash
  # Default configuration (no environment variable needed)
  # Or explicitly:
  export VS_PROFILE="local"
  ```

- **Chroma Cloud**: For production environments

  ```bash
  export VS_PROFILE="cloud"
  export CHROMA_API_KEY="your-chroma-cloud-api-key"
  ```

### Configuration

All components are configurable via YAML files:

- `configs/chunking.yml`: Chunk size, overlap, separators
- `configs/embeddings.yml`: Embedding providers and options
- `configs/vector_store.yml`: Vector store configuration

### Example Usage

```python
from src.core.loader import load_documents
from src.core.chunker import chunk_documents
from src.core.vector_store import embed_and_store, load_vector_store

# Load document
docs = load_documents("sample.txt")
# Create chunks
chunks = chunk_documents(docs)
# Generate embeddings and store
db = embed_and_store(chunks)
# Similarity search
results = db.similarity_search("your question", k=3)
```

### Tests

```bash
# Run unit and integration tests
poetry run pytest
# Ollama-dependent tests can be skipped with environment variable
SKIP_OLLAMA_TESTS=true poetry run pytest
```

### Architecture

The pipeline follows the classic RAG pattern:

```text
Documents → Chunking → Embedding → Vector Store → Retrieval → LLM → Answer
