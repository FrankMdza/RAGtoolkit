# RAG System - Google Colab Version

A Jupyter notebook implementation of the RAG system optimized for Google Colab.

## Features

- GPU-accelerated processing
- Google Drive integration
- Interactive execution
- Step-by-step implementation guide
- Performance monitoring

## Setup

1. Open the notebook in Google Colab
2. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Install dependencies:
```python
!pip install -r requirements.txt
```

## Usage

The notebook is divided into sections:

1. Environment Setup
   - GPU verification
   - Package installation
   - Drive mounting

2. Document Processing
   - Document loading
   - Text chunking
   - Error handling

3. Embedding Generation
   - Model initialization
   - Batch processing
   - GPU utilization

4. Vector Store Setup
   - ChromaDB/Pinecone configuration
   - Data persistence

5. Query Processing
   - Query embedding
   - Similarity search
   - Results processing

## Project Structure

```
colab/
├── RAG_System_Colab.ipynb  # Main notebook
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Configuration

1. Enable GPU in Colab:
   - Runtime > Change runtime type > GPU

2. Set up environment variables:
```python
import os
os.environ["PINECONE_API_KEY"] = "your_api_key"  # If using Pinecone
```

## Performance Tips

- Use larger batch sizes for embedding generation
- Monitor GPU memory usage
- Adjust chunk sizes based on document types
- Use appropriate embedding models for your needs 