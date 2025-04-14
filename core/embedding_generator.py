import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class EmbeddingGenerator:
    """Handles embedding generation for documents."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        # Warm-up run
        self.model.encode(["warmup"], batch_size=1)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Generate embeddings for a list of texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    
    def process_documents(self, documents: List[Document], batch_size: int = 32) -> torch.Tensor:
        """Process a list of documents and return their embeddings."""
        texts = [doc.page_content for doc in documents]
        return self.generate_embeddings(texts, batch_size)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension() 