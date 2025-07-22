import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock
from src.core.loader import load_documents

def test_load_txt():
    """Test loading a simple text file"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write("Hello world")
        path = f.name
    
    try:
        docs = load_documents(path)
        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["source"] == path
    finally:
        os.unlink(path)

def test_load_markdown():
    """Test loading a markdown file"""
    content = "# Title\n\nThis is markdown content."
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode='w') as f:
        f.write(content)
        path = f.name
    
    try:
        docs = load_documents(path)
        assert len(docs) == 1
        assert "Title" in docs[0].page_content
        assert "markdown content" in docs[0].page_content
    finally:
        os.unlink(path)

def test_load_file_not_found():
    """Test error handling for non-existent files"""
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_documents("non_existent_file.txt")

def test_load_empty_file():
    """Test loading an empty file"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write("")  # Empty file
        path = f.name
    
    try:
        docs = load_documents(path)
        assert len(docs) == 1
        assert docs[0].page_content == ""
    finally:
        os.unlink(path)

def test_load_large_text_file():
    """Test loading a larger text file"""
    large_content = "This is a test line.\n" * 1000  # 1000 lines
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write(large_content)
        path = f.name
    
    try:
        docs = load_documents(path)
        assert len(docs) == 1
        assert len(docs[0].page_content) > 10000  # Should be substantial
        assert "This is a test line." in docs[0].page_content
    finally:
        os.unlink(path)

def test_different_file_extensions():
    """Test that different extensions use appropriate loaders"""
    test_cases = [
        (".txt", "Text content"),
        (".md", "# Markdown content"),
    ]
    
    for ext, content in test_cases:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w') as f:
            f.write(content)
            path = f.name
        
        try:
            docs = load_documents(path)
            assert len(docs) >= 1
            assert content.strip() in docs[0].page_content or content in docs[0].page_content
        finally:
            os.unlink(path)

@patch('src.core.loader.PyPDFLoader')
def test_pdf_loader_selection(mock_pdf_loader):
    """Test that PDF files use PyPDFLoader"""
    mock_instance = MagicMock()
    mock_instance.load.return_value = [MagicMock(page_content="PDF content", metadata={"source": "test.pdf"})]
    mock_pdf_loader.return_value = mock_instance
    
    # Create a dummy PDF file (just for file existence check)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    
    try:
        docs = load_documents(path)
        mock_pdf_loader.assert_called_once_with(path)
        mock_instance.load.assert_called_once()
    finally:
        os.unlink(path)

@patch('src.core.loader.UnstructuredFileLoader')
def test_unstructured_loader_for_unknown_extension(mock_unstructured_loader):
    """Test that unknown file extensions use UnstructuredFileLoader"""
    mock_instance = MagicMock()
    mock_instance.load.return_value = [MagicMock(page_content="Unknown content", metadata={"source": "test.xyz"})]
    mock_unstructured_loader.return_value = mock_instance
    
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name
    
    try:
        docs = load_documents(path)
        mock_unstructured_loader.assert_called_once_with(path)
        mock_instance.load.assert_called_once()
    finally:
        os.unlink(path)

def test_loader_exception_handling():
    """Test error handling when loader fails"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write("test content")
        path = f.name
    
    try:
        # Mock the TextLoader to raise an exception
        with patch('src.core.loader.TextLoader') as mock_loader:
            mock_instance = MagicMock()
            mock_instance.load.side_effect = Exception("Loader failed")
            mock_loader.return_value = mock_instance
            
            with pytest.raises(Exception, match="Loader failed"):
                load_documents(path)
    finally:
        os.unlink(path)

def test_metadata_preservation():
    """Test that metadata is properly preserved"""
    content = "Test content for metadata"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write(content)
        path = f.name
    
    try:
        docs = load_documents(path)
        assert len(docs) == 1
        assert "source" in docs[0].metadata
        assert docs[0].metadata["source"] == path
    finally:
        os.unlink(path)
