from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yaml
import logging

logger = logging.getLogger(__name__)

def chunk_documents(docs: list[Document]) -> list[Document]:
    logger.info(f"Starting to chunk {len(docs)} documents")
    
    with open("configs/chunking.yml") as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Loaded chunking config: chunk_size={cfg['chunk_size']}, chunk_overlap={cfg['chunk_overlap']}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=cfg.get("separators", ["\n\n", "\n", " ", ""])
    )
    
    chunks: list[Document] = []
    total_original_chars = 0
    
    for doc_idx, doc in enumerate(docs):
        original_length = len(doc.page_content)
        total_original_chars += original_length
        
        logger.debug(f"Processing document {doc_idx + 1}/{len(docs)} (length: {original_length} chars)")
        
        texts = splitter.split_text(doc.page_content)
        doc_chunks = 0
        
        for i, chunk in enumerate(texts):
            metadata = {**doc.metadata, "chunk": i, "source_doc": doc_idx}
            chunks.append(Document(page_content=chunk, metadata=metadata))
            doc_chunks += 1
        
        logger.debug(f"Document {doc_idx + 1} split into {doc_chunks} chunks")
    
    total_chunk_chars = sum(len(chunk.page_content) for chunk in chunks)
    
    # Calculate average chunk size with robust handling of edge cases
    if not chunks:
        avg_chunk_size = 0
        avg_msg = "no chunks created"
    elif total_chunk_chars == 0:
        avg_chunk_size = 0
        avg_msg = "all chunks empty"
    else:
        avg_chunk_size = total_chunk_chars // len(chunks)
        avg_msg = f"avg chunk size: {avg_chunk_size} chars"
    
    logger.info(f"Chunking completed: {len(chunks)} chunks created from {total_original_chars} chars ({avg_msg})")
    
    return chunks
