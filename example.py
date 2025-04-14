import os
from dotenv import load_dotenv
from pipeline import RAGPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the pipeline
    pipeline = RAGPipeline(
        use_pinecone=False,  # Set to True to use Pinecone instead of ChromaDB
        persist_path="./chroma_db",  # Path to store ChromaDB data
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Process a folder of documents
    input_folder = "./input_documents"
    result = pipeline.process_folder(input_folder)
    print("Processing result:", result)
    
    # Query the system
    query = "What is the main topic of the documents?"
    results = pipeline.query(query)
    print("\nQuery results:", results)
    
    # Get system statistics
    stats = pipeline.get_stats()
    print("\nSystem statistics:", stats)

if __name__ == "__main__":
    main() 