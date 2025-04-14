from agent_models import Task, TaskType, TaskParameters, SystemConfig
from agent_implementation import RAGAgent
import logfire

def main():
    # Configure logging
    logfire.configure(
        project_name="rag_system",
        service_name="rag_agent_example",
        environment="development"
    )
    
    # Initialize the agent with custom configuration
    config = SystemConfig(
        chunk_size=500,
        chunk_overlap=100,
        embedding_model="all-mpnet-base-v2",
        batch_size=64
    )
    agent = RAGAgent(config=config)
    
    # Example 1: Process a folder of documents
    print("Example 1: Processing documents")
    folder_path = "./input_documents"
    
    # Create tasks with proper parameters
    load_task = Task(
        task_type=TaskType.DOCUMENT_LOAD,
        parameters=TaskParameters(folder_path=folder_path),
        priority=1
    )
    
    chunk_task = Task(
        task_type=TaskType.DOCUMENT_CHUNK,
        parameters=TaskParameters(),
        priority=2,
        dependencies=[load_task]
    )
    
    embedding_task = Task(
        task_type=TaskType.EMBEDDING_GENERATE,
        parameters=TaskParameters(),
        priority=3,
        dependencies=[chunk_task]
    )
    
    store_task = Task(
        task_type=TaskType.VECTOR_STORE_ADD,
        parameters=TaskParameters(),
        priority=4,
        dependencies=[embedding_task]
    )
    
    # Add tasks to the agent
    for task in [load_task, chunk_task, embedding_task, store_task]:
        agent.add_task(task)
    
    # Run the tasks
    agent.run()
    
    # Example 2: Query the system
    print("\nExample 2: Querying the system")
    query_task = Task(
        task_type=TaskType.VECTOR_STORE_QUERY,
        parameters=TaskParameters(
            query_text="What is the main topic of the documents?",
            n_results=5
        ),
        priority=1
    )
    agent.add_task(query_task)
    agent.run()
    
    # Example 3: Update configuration
    print("\nExample 3: Updating configuration")
    config_task = Task(
        task_type=TaskType.CONFIG_UPDATE,
        parameters=TaskParameters(
            config_updates={
                "chunk_size": 800,
                "chunk_overlap": 150,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        ),
        priority=1
    )
    agent.add_task(config_task)
    agent.run()
    
    # Example 4: Get system status
    print("\nExample 4: Getting system status")
    status_task = Task(
        task_type=TaskType.SYSTEM_STATUS,
        parameters=TaskParameters(),
        priority=1
    )
    agent.add_task(status_task)
    agent.run()
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(agent.metrics.dict())

if __name__ == "__main__":
    main() 