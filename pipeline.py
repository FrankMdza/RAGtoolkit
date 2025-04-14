import os
from typing import Optional, Dict, Any
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from langchain.schema import Document

class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(
        self,
        use_pinecone: bool = False,
        persist_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.vector_store = VectorStore(
            use_pinecone=use_pinecone,
            persist_path=persist_path
        )
        
        # Initialize vector store
        if use_pinecone:
            self.vector_store.setup_pinecone(
                dimension=self.embedding_generator.embedding_dimension
            )
        else:
            self.vector_store.setup_chroma()
    
    def process_folder(self, input_folder: str) -> Dict[str, Any]:
        """Process a folder of documents and store them in the vector database."""
        # 1. Load and chunk documents
        chunks = self.document_processor.process_folder(input_folder)
        
        # 2. Generate embeddings
        embeddings = self.embedding_generator.process_documents(chunks)
        
        # 3. Store in vector database
        metadatas = [chunk.metadata for chunk in chunks]
        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings.cpu().numpy().tolist(),
            metadatas=metadatas
        )
        
        return {
            "status": "success",
            "num_documents": len(chunks),
            "stats": self.vector_store.get_stats()
        }
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store with a text query."""
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embeddings(
            [query_text],
            batch_size=1
        ).cpu().numpy().tolist()[0]
        
        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        return self.vector_store.get_stats() 