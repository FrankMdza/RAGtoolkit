from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

class TaskType(Enum):
    """Types of tasks that can be performed by the agent."""
    DOCUMENT_LOAD = "document_load"
    DOCUMENT_CHUNK = "document_chunk"
    EMBEDDING_GENERATE = "embedding_generate"
    VECTOR_STORE_ADD = "vector_store_add"
    VECTOR_STORE_QUERY = "vector_store_query"
    SYSTEM_STATUS = "system_status"
    CONFIG_UPDATE = "config_update"

@dataclass
class Task:
    """Base class for all tasks."""
    task_type: TaskType
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List['Task'] = None

class RAGAgent:
    """Agent for managing RAG system operations."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = None
        self.task_queue = []
        self.config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "all-MiniLM-L6-v2",
            "use_pinecone": False,
            "persist_path": "./chroma_db"
        }
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue."""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)
    
    def execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        if task.dependencies:
            for dep in task.dependencies:
                self.execute_task(dep)
        
        if task.task_type == TaskType.DOCUMENT_LOAD:
            return self._execute_document_load(task.parameters)
        elif task.task_type == TaskType.DOCUMENT_CHUNK:
            return self._execute_document_chunk(task.parameters)
        elif task.task_type == TaskType.EMBEDDING_GENERATE:
            return self._execute_embedding_generate(task.parameters)
        elif task.task_type == TaskType.VECTOR_STORE_ADD:
            return self._execute_vector_store_add(task.parameters)
        elif task.task_type == TaskType.VECTOR_STORE_QUERY:
            return self._execute_vector_store_query(task.parameters)
        elif task.task_type == TaskType.SYSTEM_STATUS:
            return self._execute_system_status()
        elif task.task_type == TaskType.CONFIG_UPDATE:
            return self._execute_config_update(task.parameters)
    
    def _execute_document_load(self, params: Dict[str, Any]) -> List[Any]:
        """Load documents from a folder."""
        folder_path = params.get("folder_path")
        if not folder_path:
            raise ValueError("folder_path is required")
        return self.document_processor.load_documents(folder_path)
    
    def _execute_document_chunk(self, params: Dict[str, Any]) -> List[Any]:
        """Chunk documents."""
        documents = params.get("documents")
        if not documents:
            raise ValueError("documents are required")
        return self.document_processor.chunk_documents(documents)
    
    def _execute_embedding_generate(self, params: Dict[str, Any]) -> Any:
        """Generate embeddings for documents."""
        documents = params.get("documents")
        if not documents:
            raise ValueError("documents are required")
        return self.embedding_generator.process_documents(documents)
    
    def _execute_vector_store_add(self, params: Dict[str, Any]) -> None:
        """Add documents to vector store."""
        if not self.vector_store:
            self.vector_store = VectorStore(
                use_pinecone=self.config["use_pinecone"],
                persist_path=self.config["persist_path"]
            )
            if self.config["use_pinecone"]:
                self.vector_store.setup_pinecone()
            else:
                self.vector_store.setup_chroma()
        
        documents = params.get("documents")
        embeddings = params.get("embeddings")
        if not documents or not embeddings:
            raise ValueError("documents and embeddings are required")
        
        self.vector_store.add_documents(documents, embeddings)
    
    def _execute_vector_store_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        query_text = params.get("query_text")
        n_results = params.get("n_results", 5)
        if not query_text:
            raise ValueError("query_text is required")
        
        query_embedding = self.embedding_generator.generate_embeddings(
            [query_text],
            batch_size=1
        ).cpu().numpy().tolist()[0]
        
        return self.vector_store.query(query_embedding, n_results)
    
    def _execute_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = {
            "config": self.config,
            "vector_store_initialized": self.vector_store is not None
        }
        if self.vector_store:
            status.update(self.vector_store.get_stats())
        return status
    
    def _execute_config_update(self, params: Dict[str, Any]) -> None:
        """Update system configuration."""
        for key, value in params.items():
            if key in self.config:
                self.config[key] = value
    
    def run(self) -> None:
        """Execute all tasks in the queue."""
        while self.task_queue:
            task = self.task_queue.pop(0)
            try:
                result = self.execute_task(task)
                print(f"Task {task.task_type.value} completed successfully")
            except Exception as e:
                print(f"Task {task.task_type.value} failed: {str(e)}")

# Example usage:
def create_rag_pipeline_tasks(folder_path: str) -> List[Task]:
    """Create a sequence of tasks for processing a folder of documents."""
    return [
        Task(
            task_type=TaskType.DOCUMENT_LOAD,
            parameters={"folder_path": folder_path},
            priority=1
        ),
        Task(
            task_type=TaskType.DOCUMENT_CHUNK,
            parameters={},
            priority=2,
            dependencies=[Task(TaskType.DOCUMENT_LOAD, {"folder_path": folder_path})]
        ),
        Task(
            task_type=TaskType.EMBEDDING_GENERATE,
            parameters={},
            priority=3,
            dependencies=[Task(TaskType.DOCUMENT_CHUNK, {})]
        ),
        Task(
            task_type=TaskType.VECTOR_STORE_ADD,
            parameters={},
            priority=4,
            dependencies=[Task(TaskType.EMBEDDING_GENERATE, {})]
        )
    ]

def create_query_task(query_text: str, n_results: int = 5) -> Task:
    """Create a task for querying the system."""
    return Task(
        task_type=TaskType.VECTOR_STORE_QUERY,
        parameters={
            "query_text": query_text,
            "n_results": n_results
        },
        priority=1
    ) 