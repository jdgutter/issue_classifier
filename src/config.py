import os
from pathlib import Path

# Resolve the absolute path of the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """
    Centralized configuration engine. 
    Enforces absolute path resolution and environment variable overrides 
    to support seamless local, CI/CD, and cloud deployments.
    """
    # --- Data & Storage Paths ---
    DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", BASE_DIR / "data" / "raw"))
    DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", BASE_DIR / "data" / "processed"))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
    
    # Ingestion Source
    RAW_CSV_PATH = DATA_RAW_DIR / "smaller.csv"
    
    # Model Artifact Names
    CLASSIFIER_PATH = MODELS_DIR / "pipeline.joblib"
    RANKER_MODEL_PATH = MODELS_DIR / "ranking_model.pt"
    FAISS_INDEX_PATH = MODELS_DIR / "index.faiss"
    FAISS_METADATA_PATH = MODELS_DIR / "issues_metadata.json"

    # --- Recommendation Pipeline Settings ---
    # Default candidate pool sizes
    K_RETRIEVAL_DEFAULT = 50
    K_RECOMMENDATIONS_DEFAULT = 5

    # Neural Network Hyperparameters
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384
    NUM_CATEGORICAL_TAGS = 10
    CATEGORICAL_TAG_EMBED_DIM = 8

    # --- API Serving Layer Settings ---
    API_TITLE = "GitHub Issue Classifier & Recommender API"
    API_VERSION = "2.0.0"

# Instantiated singleton config to import across the codebase
settings = Settings()
