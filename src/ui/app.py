from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.core.ingestion import ingest_files, store_uploaded_file
from src.core.rag import answer
from src.core.vector_store import delete_by_source

load_dotenv()
st.set_page_config(page_title="Naive RAG", layout="wide")


def _init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("upload_summary", None)
    st.session_state.setdefault("top_k", 8)
    st.session_state.setdefault("processed_upload_signatures", set())
    st.session_state.setdefault("stored_uploads", {})


def _get_processed_signatures() -> set[str]:
    processed = st.session_state.get("processed_upload_signatures")
    if isinstance(processed, set):
        return processed
    restored = set(processed or [])
    st.session_state.processed_upload_signatures = restored
    return restored


def _get_stored_uploads() -> Dict[str, str]:
    stored = st.session_state.get("stored_uploads")
    if isinstance(stored, dict):
        return stored
    restored = dict(stored or {})
    st.session_state.stored_uploads = restored
    return restored


def _remove_missing_uploads(current_signatures: set[str]) -> None:
    stored_uploads = _get_stored_uploads()
    processed = _get_processed_signatures()
    removed = [
        signature for signature in stored_uploads if signature not in current_signatures
    ]
    for signature in removed:
        path_str = stored_uploads.get(signature)
        file_path = Path(path_str) if path_str else None
        deletion_error = None
        if file_path:
            try:
                file_path.unlink(missing_ok=True)
            except Exception as exc:
                deletion_error = exc
        if deletion_error:
            st.sidebar.warning(f"Failed to delete '{file_path.name}': {deletion_error}")
        else:
            stored_uploads.pop(signature, None)
            if file_path:
                delete_by_source(str(file_path))
        processed.discard(signature)


def _collect_upload_payloads(uploads) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    if not uploads:
        return payloads

    seen_signatures: set[str] = set()
    for upload in uploads:
        if upload.size == 0:
            continue
        data = upload.getvalue()
        signature = f"{upload.name}:{hashlib.md5(data).hexdigest()}"
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        payloads.append(
            {
                "name": upload.name,
                "data": data,
                "signature": signature,
            }
        )
    return payloads


def _ingest_payloads(payloads: List[Dict[str, Any]]):
    saved_paths: List[Path] = []
    stored_uploads = _get_stored_uploads()
    for payload in payloads:
        stored = store_uploaded_file(payload["data"], payload["name"])
        saved_paths.append(stored)
        stored_uploads[payload["signature"]] = str(stored)

    summary = ingest_files(saved_paths)
    st.session_state.upload_summary = summary

    processed = _get_processed_signatures()
    for payload, report in zip(payloads, summary.files):
        if report.status != "failed":
            processed.add(payload["signature"])

    return summary


def _format_ingestion_message(payloads: List[Dict[str, Any]], summary) -> str:
    status_icon = {
        "success": "✅",
        "skipped": "⚠️",
        "failed": "❌",
    }
    lines = ["📄 New documents processed:"]
    for payload, report in zip(payloads, summary.files):
        icon = status_icon.get(report.status, "•")
        message = f"{icon} {payload['name']}"
        if report.error:
            message += f" – {report.error}"
        lines.append(message)

    lines.append(
        f"Documents: {summary.total_documents} • Chunks: {summary.total_chunks}"
    )
    if summary.vector_store_error:
        lines.append(f"⚠️ Vector store: {summary.vector_store_error}")
    return "\n".join(lines)


def _render_sidebar() -> List[Dict[str, Any]]:
    st.sidebar.header("Knowledge Base")
    st.sidebar.write(
        "Upload documents to add them to the vector store. "
        "Supported types include PDF, TXT, MD, DOCX, and CSV."
    )

    uploads = st.sidebar.file_uploader(
        "Select documents",
        type=["pdf", "txt", "md", "docx", "csv"],
        accept_multiple_files=True,
    )

    payloads = _collect_upload_payloads(uploads)
    current_signatures = {payload["signature"] for payload in payloads}
    _remove_missing_uploads(current_signatures)

    if uploads:
        st.sidebar.caption(
            "Documents will be automatically processed on the next chat request."
        )

    if st.session_state.get("upload_summary"):
        with st.sidebar.expander("Last ingestion result", expanded=True):
            _render_ingestion_summary(st.session_state.upload_summary)

    if st.sidebar.button("Clear chat history"):
        st.session_state.chat_history = []

    st.session_state.top_k = st.sidebar.slider(
        "Number of context chunks (top_k)",
        min_value=1,
        max_value=20,
        value=int(st.session_state.top_k),
        help="How many chunks to retrieve per question from the vector store.",
    )

    return payloads


def _render_ingestion_feedback(summary) -> None:
    if summary.vector_store_error:
        st.sidebar.error(f"Failed to update vector store: {summary.vector_store_error}")
    elif summary.total_chunks:
        st.sidebar.success(
            f"Processed {summary.total_documents} document(s) and "
            f"{summary.total_chunks} chunk(s)."
        )
    else:
        st.sidebar.warning(
            "No chunks created — check the documents or chunking configuration."
        )

    failed = [item for item in summary.files if item.status == "failed"]
    skipped = [item for item in summary.files if item.status == "skipped"]
    if failed:
        st.sidebar.error(
            "Errors for the following files:"
            + "\n"
            + "\n".join(f"- {Path(item.path).name}: {item.error}" for item in failed)
        )
    if skipped:
        st.sidebar.warning(
            "Skipped files:"
            + "\n"
            + "\n".join(f"- {Path(item.path).name}: {item.error}" for item in skipped)
        )


def _render_ingestion_summary(summary) -> None:
    if not summary:
        st.write("No ingestion data available.")
        return

    st.write(
        f"Documents: {summary.total_documents} | Chunks: {summary.total_chunks} | "
        f"Failures: {summary.failures}"
    )
    for item in summary.files:
        status_icon = {
            "success": "✅",
            "skipped": "⚠️",
            "failed": "❌",
        }.get(item.status, "•")
        description = f"{status_icon} {Path(item.path).name}"
        if item.chunks:
            description += f" – {item.chunks} Chunks"
        if item.error:
            description += f" ({item.error})"
        st.markdown(description)


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if role == "assistant" and message.get("sources"):
                st.caption("Sources")
                for source in message["sources"]:
                    src = source.get("source", "unknown")
                    chunk = source.get("chunk")
                    label = f"{src}"
                    if chunk is not None:
                        label += f" (Chunk {chunk})"
                    st.markdown(f"- {label}")


def _handle_user_query(current_uploads: List[Dict[str, Any]]) -> None:
    prompt = st.chat_input("Ask a question…")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    processed = _get_processed_signatures()
    new_payloads = [
        payload for payload in current_uploads if payload["signature"] not in processed
    ]

    if new_payloads:
        try:
            with st.chat_message("assistant"):
                st.markdown("New documents detected. Processing...")
                with st.spinner("Creating embeddings and updating the vector store..."):
                    summary = _ingest_payloads(new_payloads)
                ingestion_text = _format_ingestion_message(new_payloads, summary)
                st.markdown(ingestion_text)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ingestion_text, "sources": []}
            )
        except Exception as exc:
            error_text = f"Error while processing new documents: {exc}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_text, "sources": []}
            )
            with st.chat_message("assistant"):
                st.error(error_text)
            return

    try:
        with st.spinner("Generating answer..."):
            rag_response: Dict[str, Any] = answer(prompt, k=int(st.session_state.top_k))
    except Exception as exc:
        error_text = f"Query failed: {exc}"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error_text, "sources": []}
        )
        with st.chat_message("assistant"):
            st.error(error_text)
        return

    content = rag_response.get("answer", "(No answer)")
    sources = rag_response.get("sources", []) or []

    st.session_state.chat_history.append(
        {"role": "assistant", "content": content, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(content)
        if sources:
            st.caption("Sources")
            for source in sources:
                src = source.get("source", "unknown")
                chunk = source.get("chunk")
                label = f"{src}"
                if chunk is not None:
                    label += f" (Chunk {chunk})"
                st.markdown(f"- {label}")


def main() -> None:
    _init_session_state()
    st.title("Naive RAG")
    st.write(
        "Ask questions about the knowledge base. Upload new documents in the sidebar "
        "to expand the vector store."
    )

    current_uploads = _render_sidebar()
    _render_chat_history()
    _handle_user_query(current_uploads)


if __name__ == "__main__":
    main()
