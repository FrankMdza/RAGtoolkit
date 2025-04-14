import os
from typing import List, Dict, Any
from langchain.document_loaders import (
    PyMuPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """Handles document loading and chunking operations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_map = {
            '.pdf': PyMuPDFLoader,
            '.csv': CSVLoader,
            '.txt': TextLoader,
            '.md': TextLoader
        }
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from a folder with error handling."""
        docs = []
        for file in os.listdir(folder_path):
            try:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.document_map:
                    loader = self.document_map[ext](f"{folder_path}/{file}")
                    docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {file}: {str(e)}")
        return docs
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return splitter.split_documents(docs)
    
    def process_folder(self, folder_path: str) -> List[Document]:
        """Complete document processing pipeline."""
        docs = self.load_documents(folder_path)
        return self.chunk_documents(docs) 