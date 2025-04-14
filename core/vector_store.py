import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import pinecone
from langchain.schema import Document

class VectorStore:
    """Handles vector database operations for both ChromaDB and Pinecone."""
    
    def __init__(self, use_pinecone: bool = False, persist_path: Optional[str] = None):
        self.use_pinecone = use_pinecone
        self.persist_path = persist_path
        self.collection = None
        self.index = None
        
        if use_pinecone:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is required")
            pinecone.init(api_key=api_key)
        else:
            if not persist_path:
                raise ValueError("persist_path is required for ChromaDB")
    
    def setup_chroma(self, collection_name: str = "docs") -> None:
        """Set up ChromaDB collection."""
        if self.use_pinecone:
            raise ValueError("Cannot setup ChromaDB when using Pinecone")
            
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_path
        )
        client = chromadb.Client(chroma_settings)
        self.collection = client.create_collection(collection_name)
    
    def setup_pinecone(self, index_name: str = "docs", dimension: int = 384) -> None:
        """Set up Pinecone index."""
        if not self.use_pinecone:
            raise ValueError("Cannot setup Pinecone when using ChromaDB")
            
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension)
        self.index = pinecone.Index(index_name)
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], 
                     metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents to the vector store."""
        ids = [f"vec_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        
        if self.use_pinecone:
            if not self.index:
                raise ValueError("Pinecone index not initialized")
            self.index.upsert(vectors=zip(ids, embeddings, metadatas or [{}] * len(documents)))
        else:
            if not self.collection:
                raise ValueError("ChromaDB collection not initialized")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store."""
        if self.use_pinecone:
            if not self.index:
                raise ValueError("Pinecone index not initialized")
            return self.index.query(vector=query_embedding, top_k=n_results)
        else:
            if not self.collection:
                raise ValueError("ChromaDB collection not initialized")
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.use_pinecone:
            if not self.index:
                raise ValueError("Pinecone index not initialized")
            return {"status": "healthy", "index_stats": self.index.describe_index_stats()}
        else:
            if not self.collection:
                raise ValueError("ChromaDB collection not initialized")
            return {"status": "healthy", "count": self.collection.count()} 