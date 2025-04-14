from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
import logfire
from datetime import datetime

class TaskType(str, Enum):
    """Types of tasks that can be performed by the agent."""
    DOCUMENT_LOAD = "document_load"
    DOCUMENT_CHUNK = "document_chunk"
    EMBEDDING_GENERATE = "embedding_generate"
    VECTOR_STORE_ADD = "vector_store_add"
    VECTOR_STORE_QUERY = "vector_store_query"
    SYSTEM_STATUS = "system_status"
    CONFIG_UPDATE = "config_update"

class TaskParameters(BaseModel):
    """Base model for task parameters."""
    folder_path: Optional[str] = None
    documents: Optional[List[Any]] = None
    embeddings: Optional[List[List[float]]] = None
    query_text: Optional[str] = None
    n_results: Optional[int] = 5
    config_updates: Optional[Dict[str, Any]] = None

class Task(BaseModel):
    """Model for a task in the RAG system."""
    task_type: TaskType
    parameters: TaskParameters
    priority: int = Field(default=1, ge=1, le=5)
    dependencies: Optional[List['Task']] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="pending")
    error: Optional[str] = None

    @validator('dependencies')
    def validate_dependencies(cls, v, values):
        if v is not None:
            for dep in v:
                if dep.task_type == values.get('task_type'):
                    raise ValueError("Task cannot depend on itself")
        return v

class SystemConfig(BaseModel):
    """Model for system configuration."""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    use_pinecone: bool = Field(default=False)
    persist_path: str = Field(default="./chroma_db")
    batch_size: int = Field(default=32, ge=1, le=128)

class TaskResult(BaseModel):
    """Model for task execution results."""
    task_id: str
    task_type: TaskType
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SystemStatus(BaseModel):
    """Model for system status."""
    config: SystemConfig
    vector_store_initialized: bool
    total_tasks_executed: int
    successful_tasks: int
    failed_tasks: int
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    current_memory_usage: float
    gpu_available: bool
    gpu_memory_usage: Optional[float] = None

class RAGMetrics(BaseModel):
    """Model for RAG system metrics."""
    documents_processed: int = 0
    chunks_generated: int = 0
    embeddings_generated: int = 0
    queries_executed: int = 0
    average_query_time: float = 0.0
    average_embedding_time: float = 0.0
    total_errors: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Initialize Logfire
logfire.configure(
    project_name="rag_system",
    service_name="rag_agent",
    environment="development"
)

# Create loggers
task_logger = logfire.get_logger("task_execution")
system_logger = logfire.get_logger("system_status")
error_logger = logfire.get_logger("error_tracking")
metrics_logger = logfire.get_logger("system_metrics") 