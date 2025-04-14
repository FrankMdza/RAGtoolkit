import os
import time
import psutil
import torch
from typing import List, Dict, Any, Optional
from uuid import uuid4
from .agent_models import (
    Task, TaskType, TaskParameters, SystemConfig, TaskResult,
    SystemStatus, RAGMetrics, task_logger, system_logger,
    error_logger, metrics_logger
)
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

class RAGAgent:
    """Agent for managing RAG system operations with monitoring."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.embedding_model
        )
        self.vector_store = None
        self.task_queue: List[Task] = []
        self.metrics = RAGMetrics()
        self._initialize_vector_store()
        
        system_logger.info("RAGAgent initialized", extra={
            "config": self.config.dict(),
            "gpu_available": torch.cuda.is_available()
        })
    
    def _initialize_vector_store(self) -> None:
        """Initialize the vector store based on configuration."""
        try:
            self.vector_store = VectorStore(
                use_pinecone=self.config.use_pinecone,
                persist_path=self.config.persist_path
            )
            if self.config.use_pinecone:
                self.vector_store.setup_pinecone()
            else:
                self.vector_store.setup_chroma()
            system_logger.info("Vector store initialized successfully")
        except Exception as e:
            error_logger.error("Failed to initialize vector store", extra={"error": str(e)})
            raise
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue with validation."""
        try:
            task.validate()
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda x: (x.priority, x.created_at), reverse=True)
            task_logger.info("Task added to queue", extra={
                "task_type": task.task_type,
                "priority": task.priority
            })
        except Exception as e:
            error_logger.error("Failed to add task", extra={
                "error": str(e),
                "task_type": task.task_type
            })
            raise
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            "memory_usage": memory_info.rss / 1024**2,  # MB
            "gpu_memory": gpu_memory,
            "cpu_percent": process.cpu_percent()
        }
    
    def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with monitoring."""
        task_id = str(uuid4())
        start_time = time.time()
        
        try:
            # Execute dependencies first
            if task.dependencies:
                for dep in task.dependencies:
                    self.execute_task(dep)
            
            # Execute the task
            task_logger.info("Executing task", extra={
                "task_id": task_id,
                "task_type": task.task_type
            })
            
            result = self._execute_task_internal(task)
            duration = time.time() - start_time
            
            # Update metrics
            self._update_metrics(task.task_type, duration)
            
            return TaskResult(
                task_id=task_id,
                task_type=task.task_type,
                status="success",
                result=result,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_logger.error("Task execution failed", extra={
                "task_id": task_id,
                "task_type": task.task_type,
                "error": str(e)
            })
            self.metrics.total_errors += 1
            
            return TaskResult(
                task_id=task_id,
                task_type=task.task_type,
                status="failed",
                error=str(e),
                duration=duration
            )
    
    def _execute_task_internal(self, task: Task) -> Any:
        """Internal task execution logic."""
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
    
    def _update_metrics(self, task_type: TaskType, duration: float) -> None:
        """Update system metrics based on task execution."""
        if task_type == TaskType.DOCUMENT_LOAD:
            self.metrics.documents_processed += 1
        elif task_type == TaskType.DOCUMENT_CHUNK:
            self.metrics.chunks_generated += 1
        elif task_type == TaskType.EMBEDDING_GENERATE:
            self.metrics.embeddings_generated += 1
            self.metrics.average_embedding_time = (
                (self.metrics.average_embedding_time * (self.metrics.embeddings_generated - 1) + duration) /
                self.metrics.embeddings_generated
            )
        elif task_type == TaskType.VECTOR_STORE_QUERY:
            self.metrics.queries_executed += 1
            self.metrics.average_query_time = (
                (self.metrics.average_query_time * (self.metrics.queries_executed - 1) + duration) /
                self.metrics.queries_executed
            )
        
        self.metrics.last_updated = datetime.utcnow()
        metrics_logger.info("Metrics updated", extra=self.metrics.dict())
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        metrics = self._get_system_metrics()
        
        return SystemStatus(
            config=self.config,
            vector_store_initialized=self.vector_store is not None,
            total_tasks_executed=len(self.task_queue),
            successful_tasks=self.metrics.documents_processed,
            failed_tasks=self.metrics.total_errors,
            current_memory_usage=metrics["memory_usage"],
            gpu_available=torch.cuda.is_available(),
            gpu_memory_usage=metrics["gpu_memory"]
        )
    
    def run(self) -> None:
        """Execute all tasks in the queue with monitoring."""
        while self.task_queue:
            task = self.task_queue.pop(0)
            result = self.execute_task(task)
            
            if result.status == "success":
                system_logger.info("Task completed successfully", extra={
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "duration": result.duration
                })
            else:
                error_logger.error("Task failed", extra={
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "error": result.error
                })
    
    # Task execution methods (similar to before but with Pydantic models)
    def _execute_document_load(self, params: TaskParameters) -> List[Any]:
        if not params.folder_path:
            raise ValueError("folder_path is required")
        return self.document_processor.load_documents(params.folder_path)
    
    def _execute_document_chunk(self, params: TaskParameters) -> List[Any]:
        if not params.documents:
            raise ValueError("documents are required")
        return self.document_processor.chunk_documents(params.documents)
    
    def _execute_embedding_generate(self, params: TaskParameters) -> Any:
        if not params.documents:
            raise ValueError("documents are required")
        return self.embedding_generator.process_documents(params.documents)
    
    def _execute_vector_store_add(self, params: TaskParameters) -> None:
        if not params.documents or not params.embeddings:
            raise ValueError("documents and embeddings are required")
        self.vector_store.add_documents(params.documents, params.embeddings)
    
    def _execute_vector_store_query(self, params: TaskParameters) -> Dict[str, Any]:
        if not params.query_text:
            raise ValueError("query_text is required")
        query_embedding = self.embedding_generator.generate_embeddings(
            [params.query_text],
            batch_size=1
        ).cpu().numpy().tolist()[0]
        return self.vector_store.query(query_embedding, params.n_results)
    
    def _execute_system_status(self) -> Dict[str, Any]:
        return self.get_system_status().dict()
    
    def _execute_config_update(self, params: TaskParameters) -> None:
        if not params.config_updates:
            raise ValueError("config_updates are required")
        for key, value in params.config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value) 