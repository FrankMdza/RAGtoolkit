# Core RAG System

The core implementation of the RAG (Retrieval-Augmented Generation) system.

## Features

- Document processing (PDF, CSV, TXT, MD)
- Text chunking
- Embedding generation
- Vector store integration (ChromaDB/Pinecone)
- Query processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from document_processor import DocumentProcessor
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore

# Initialize components
processor = DocumentProcessor()
generator = EmbeddingGenerator()
store = VectorStore()

# Process documents
docs = processor.load_documents("./input_documents")
chunks = processor.chunk_documents(docs)

# Generate embeddings
embeddings = generator.process_documents(chunks)

# Store in vector database
store.add_documents(chunks, embeddings)

# Query
results = store.query("What is the main topic?")
```

## Project Structure

```
core/
├── document_processor.py  # Document loading and chunking
├── embedding_generator.py # Embedding generation
├── vector_store.py       # Vector database operations
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Configuration

Create a `.env` file with:
```
PINECONE_API_KEY=your_api_key  # Only required if using Pinecone
``` 