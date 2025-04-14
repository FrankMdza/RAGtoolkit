# RAG System - Agent Version

An agent-based implementation of the RAG system with monitoring and task management.

## Features

- Task-based architecture
- Pydantic models for type safety
- Logfire integration for monitoring
- System metrics tracking
- Error handling and recovery
- Configurable task priorities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from agent_models import Task, TaskType, TaskParameters, SystemConfig
from agent_implementation import RAGAgent

# Initialize agent with configuration
config = SystemConfig(
    chunk_size=500,
    chunk_overlap=100,
    embedding_model="all-mpnet-base-v2"
)
agent = RAGAgent(config=config)

# Create and add tasks
task = Task(
    task_type=TaskType.DOCUMENT_LOAD,
    parameters=TaskParameters(folder_path="./input_documents"),
    priority=1
)
agent.add_task(task)

# Run tasks
agent.run()
```

## Project Structure

```
agent/
├── agent_models.py       # Pydantic models
├── agent_implementation.py # Agent implementation
├── agent_example.py     # Usage examples
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## Monitoring

The system uses Logfire for monitoring:

- Task execution logs
- System status updates
- Error tracking
- Performance metrics

## Task Types

1. Document Loading
2. Document Chunking
3. Embedding Generation
4. Vector Store Operations
5. System Status
6. Configuration Updates

## Configuration

Create a `.env` file with:
```
PINECONE_API_KEY=your_api_key  # If using Pinecone
LOGFIRE_API_KEY=your_api_key   # For monitoring
```

## Metrics

The system tracks:
- Documents processed
- Chunks generated
- Embeddings created
- Query execution times
- Error rates
- System resource usage

## Error Handling

- Task validation
- Dependency checking
- Error recovery
- Status reporting
- Metrics tracking 