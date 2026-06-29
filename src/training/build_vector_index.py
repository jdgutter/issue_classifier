import os
from src.config import settings
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex
from src.validate import validate_github_issues

def main():
    csv_file = str(settings.RAW_CSV_PATH)
    if not os.path.exists(csv_file):
        print(f"Error: Raw CSV data file not found at {csv_file}")
        return

    print(f"Loading and validating issues from {csv_file}...")
    issues = list(validate_github_issues(csv_file))
    print(f"Successfully loaded {len(issues)} issues.")

    print("Initializing embedder...")
    embedder = IssueEmbedder()

    index_path = str(settings.FAISS_INDEX_PATH)
    metadata_path = str(settings.FAISS_METADATA_PATH)
    print(f"Building FAISS vector index at {index_path}...")
    
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    vector_index.build(issues, embedder)
    print(f"Vector index built and saved successfully to {index_path}!")

if __name__ == "__main__":
    main()
