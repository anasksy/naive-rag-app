import os
import logging
from typing import Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_documents(path: str) -> list[Document]:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()

    logger.info(f"Loading document: {path} (size: {file_size} bytes, type: {ext})")

    loader: Union[PyPDFLoader, TextLoader, UnstructuredFileLoader]
    if ext == ".pdf":
        logger.debug("Using PyPDFLoader for PDF file")
        loader = PyPDFLoader(path)
    elif ext in (".txt", ".md"):
        logger.debug(f"Using TextLoader for {ext} file")
        loader = TextLoader(path)
    else:
        logger.debug(f"Using UnstructuredFileLoader for {ext} file")
        loader = UnstructuredFileLoader(path)

    try:
        docs = loader.load()
        total_chars = sum(len(doc.page_content) for doc in docs)
        logger.info(
            f"Successfully loaded {len(docs)} pages with {total_chars} total characters"
        )

        # Log metadata for each document
        for i, doc in enumerate(docs):
            logger.debug(
                f"Page {i + 1}: {len(doc.page_content)} chars, metadata: {doc.metadata}"
            )

        return docs
    except Exception as e:
        logger.error(f"Failed to load document {path}: {str(e)}")
        raise
