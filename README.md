# RAG System

A modular implementation of a Retrieval-Augmented Generation (RAG) system with multiple deployment options.

## Components

### 1. Core RAG System
The base implementation of the RAG system with essential components:
- Document processing
- Embedding generation
- Vector store integration
- Query processing

[View Core Documentation](./core/README.md)

### 2. Google Colab Version
A Jupyter notebook implementation optimized for Google Colab:
- GPU acceleration
- Interactive execution
- Step-by-step guide
- Performance monitoring

[View Colab Documentation](./colab/README.md)

### 3. Agent-Based Version
An advanced implementation with monitoring and task management:
- Pydantic models
- Logfire integration
- Task-based architecture
- System metrics

[View Agent Documentation](./agent/README.md)

## Quick Start

Choose the implementation that best suits your needs:

1. For basic usage:
```bash
cd core
pip install -r requirements.txt
python example.py
```

2. For Colab usage:
- Open `colab/RAG_System_Colab.ipynb` in Google Colab
- Follow the notebook instructions

3. For agent-based usage:
```bash
cd agent
pip install -r requirements.txt
python agent_example.py
```

4. For Docker usage:
```bash
# Create a .env file with your API keys
echo "PINECONE_API_KEY=your_api_key" > .env
echo "LOGFIRE_API_KEY=your_api_key" >> .env

# Build and start the containers
docker-compose up --build

# To run in detached mode
docker-compose up -d

# To stop the containers
docker-compose down
```

## Project Structure

```
RAGsys.py/
├── core/           # Core RAG implementation
├── colab/          # Google Colab version
├── agent/          # Agent-based version
├── docker-compose.yml  # Docker orchestration
├── .dockerignore  # Docker ignore rules
└── README.md      # This file
```

## Features

- Multiple document format support (PDF, CSV, TXT, MD)
- Configurable text chunking
- GPU-accelerated embedding generation
- Vector store options (ChromaDB/Pinecone)
- Monitoring and metrics
- Error handling
- Task management

## Requirements

Each component has its own requirements file. See the respective directories for details.

## Configuration

Create a `.env` file in the root directory with:
```
PINECONE_API_KEY=your_api_key  # If using Pinecone
LOGFIRE_API_KEY=your_api_key   # For agent monitoring
```

## Docker Configuration

The Docker setup includes:

1. Optimized base images using `python:3.11-slim`
2. Resource limits for CPU and memory
3. Volume mounts for persistent data
4. Environment variable management
5. Automatic restart policies
6. Security best practices (non-root user)

To customize resource limits, modify the `docker-compose.yml` file:

```yaml
deploy:
  resources:
    limits:
      cpus: '1'    # Adjust CPU cores
      memory: 2G   # Adjust memory limit
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 