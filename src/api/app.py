from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import torch
from src.schema import GithubIssue
from src.config import settings
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex
from src.retrieval import CandidateRetriever
from src.recommender import TwoStageRecommender

# Dictionary to hold our loaded models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the serialized classifier model into memory on app startup.
    print(f"Loading model pipeline from {settings.CLASSIFIER_PATH} into memory...")
    ml_models["pipeline"] = joblib.load(settings.CLASSIFIER_PATH)
    
    # Load recommendation components
    print("Loading Two-Stage Recommender components...")
    embedder = IssueEmbedder()
    vector_index = IssueVectorIndex()
    
    try:
        vector_index.load()
        retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
        
        # Load Z-score scaling stats from the PyTorch ranking checkpoint
        checkpoint = torch.load(settings.RANKER_MODEL_PATH, map_location="cpu")
        scaling_stats = checkpoint.get("scaling_stats")
        
        onnx_model_path = str(settings.MODELS_DIR / "ranking_model.onnx")
        
        # Instantiate Recommender with the native C++ engine
        ml_models["recommender"] = TwoStageRecommender(
            retriever=retriever,
            scaling_stats=scaling_stats,
            native_engine_path=onnx_model_path
        )
        print("Two-Stage Recommender with C++ native wrapper loaded successfully.")
    except Exception as e:
        print(f"Error loading Two-Stage Recommender: {e}. Recommender endpoint will be unavailable.")
        
    yield
    # Clean up resources on shutdown
    ml_models.clear()

# Initialize the FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description="An API serving layer that classifies and recommends GitHub issues.",
    version=settings.API_VERSION,
    lifespan=lifespan
)

@app.post("/predict")
def predict_issue(issue: GithubIssue):
    """
    Accepts a JSON payload strictly formatted to the GithubIssue schema.
    Returns a predicted classification label.
    """
    # Pass the validated Pydantic object to the model pipeline to get a real prediction.
    prediction = ml_models["pipeline"].predict([issue])
    
    return {
        "issue_title": issue.issue_title,
        "predicted_label": prediction[0],
        "status": "success"
    }

@app.post("/recommend")
def recommend_issues(query: str, k: int = 5):
    """
    Accepts a search query and returns the top k recommended GitHub issues
    scored using the native C++ ranking engine.
    """
    recommender = ml_models.get("recommender")
    if not recommender:
        return {"status": "error", "message": "Recommender is not initialized."}
        
    recommendations = recommender.recommend(
        query=query,
        k_retrieval=settings.K_RETRIEVAL_DEFAULT,
        k_recommendations=k
    )
    
    results = []
    for issue, score in recommendations:
        results.append({
            "issue_url": issue.issue_url,
            "issue_title": issue.issue_title,
            "score": float(score)
        })
        
    return {
        "query": query,
        "recommendations": results,
        "status": "success"
    }
