from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from uuid import uuid4

from langchain_core.documents import Document

from src.core.chunker import chunk_documents
from src.core.loader import load_documents
from src.core.vector_store import embed_and_store

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = Path("data/uploads")


@dataclass
class FileIngestionResult:
    path: str
    documents: int = 0
    chunks: int = 0
    status: str = "pending"
    error: str | None = None


@dataclass
class IngestionSummary:
    files: List[FileIngestionResult]
    total_documents: int
    total_chunks: int
    failures: int
    vector_store_error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.failures == 0
            and self.vector_store_error is None
            and self.total_chunks > 0
        )


def ensure_upload_dir(upload_dir: Path | None = None) -> Path:
    target = DEFAULT_UPLOAD_DIR if upload_dir is None else upload_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def store_uploaded_file(
    data: bytes, filename: str, upload_dir: Path | None = None
) -> Path:
    directory = ensure_upload_dir(upload_dir)
    safe_name = Path(filename or "").name
    if not safe_name:
        safe_name = f"upload-{uuid4().hex}"
    destination = directory / safe_name
    if destination.exists():
        destination = (
            directory / f"{destination.stem}-{uuid4().hex[:8]}{destination.suffix}"
        )
    with destination.open("wb") as handle:
        handle.write(data)
    logger.info("Stored upload '%s' to %s", filename, destination)
    return destination


def _process_file(path: Path) -> Tuple[FileIngestionResult, List[Document]]:
    report = FileIngestionResult(path=str(path.resolve()))
    try:
        docs = load_documents(str(path))
        report.documents = len(docs)
        if not docs:
            report.status = "skipped"
            report.error = "Loader returned no documents"
            return report, []

        chunks = chunk_documents(docs)
        report.chunks = len(chunks)
        if not chunks:
            report.status = "skipped"
            report.error = "Chunker produced no chunks"
            return report, []

        report.status = "success"
        return report, chunks
    except Exception as exc:
        logger.exception("Failed to ingest file: %s", path)
        report.status = "failed"
        report.error = str(exc)
        return report, []


def _persist_chunks(chunks: List[Document]) -> str | None:
    if not chunks:
        logger.info("No chunks to persist; skipping vector store update")
        return None

    try:
        logger.info("Persisting %d chunks to vector store", len(chunks))
        embed_and_store(chunks)
    except Exception as exc:
        logger.exception("Failed to update vector store")
        return str(exc)
    return None


def ingest_files(paths: Iterable[str | Path]) -> IngestionSummary:
    reports: List[FileIngestionResult] = []
    collected_chunks: List[Document] = []

    for raw_path in paths:
        report, chunks = _process_file(Path(raw_path))
        reports.append(report)
        if report.status == "success":
            collected_chunks.extend(chunks)

    vector_store_error = _persist_chunks(collected_chunks)

    total_documents = sum(item.documents for item in reports)
    total_chunks = sum(item.chunks for item in reports)
    failures = sum(1 for item in reports if item.status == "failed")

    return IngestionSummary(
        files=reports,
        total_documents=total_documents,
        total_chunks=total_chunks,
        failures=failures,
        vector_store_error=vector_store_error,
    )


def ingest_directory(directory: str | Path, pattern: str = "*") -> IngestionSummary:
    directory_path = Path(directory)
    candidates = sorted(directory_path.glob(pattern))
    return ingest_files(candidate for candidate in candidates if candidate.is_file())


__all__ = [
    "DEFAULT_UPLOAD_DIR",
    "FileIngestionResult",
    "IngestionSummary",
    "ensure_upload_dir",
    "store_uploaded_file",
    "ingest_files",
    "ingest_directory",
]
