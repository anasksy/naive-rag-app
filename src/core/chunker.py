from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yaml
import logging

logger = logging.getLogger(__name__)

def chunk_documents(docs: list[Document]) -> list[Document]:
    logger.info(f"Starting to chunk {len(docs)} documents")
    
    cfg = yaml.safe_load(open("configs/chunking.yml"))
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
    logger.info(f"Chunking completed: {len(chunks)} chunks created from {total_original_chars} chars (avg chunk size: {total_chunk_chars // len(chunks) if chunks else 0} chars)")
    
    return chunks
