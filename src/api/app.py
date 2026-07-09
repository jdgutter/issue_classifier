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
        import os
        from src.ranking import IssueRankingModel

        vector_index.load()
        retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
        
        # Load Z-score scaling stats from the PyTorch ranking checkpoint
        checkpoint = torch.load(settings.RANKER_MODEL_PATH, map_location="cpu")
        scaling_stats = checkpoint.get("scaling_stats")
        
        # 1. Instantiate native unquantized C++ engine
        onnx_model_path = str(settings.MODELS_DIR / "ranking_model.onnx")
        ml_models["recommender_native"] = TwoStageRecommender(
            retriever=retriever,
            scaling_stats=scaling_stats,
            native_engine_path=onnx_model_path
        )
        print("Two-Stage Recommender with C++ native wrapper (unquantized) loaded successfully.")

        # 2. Instantiate native quantized C++ engine if available
        quantized_onnx_path = str(settings.MODELS_DIR / "ranking_model_quantized.onnx")
        if os.path.exists(quantized_onnx_path):
            ml_models["recommender_quantized"] = TwoStageRecommender(
                retriever=retriever,
                scaling_stats=scaling_stats,
                native_engine_path=quantized_onnx_path
            )
            print("Two-Stage Recommender with C++ native wrapper (quantized) loaded successfully.")
        else:
            print(f"Quantized ONNX model not found at {quantized_onnx_path}, quantized C++ recommender will be unavailable.")

        # 3. Instantiate pure Python (PyTorch) engine
        ranker_model = IssueRankingModel(
            num_tags=checkpoint.get("num_tags", settings.NUM_CATEGORICAL_TAGS),
            tag_embed_dim=checkpoint.get("tag_embed_dim", settings.CATEGORICAL_TAG_EMBED_DIM),
            embedding_dim=checkpoint.get("embedding_dim", settings.EMBEDDING_DIMENSION)
        )
        ranker_model.load_state_dict(checkpoint["model_state_dict"])
        ranker_model.eval()

        ml_models["recommender_python"] = TwoStageRecommender(
            retriever=retriever,
            ranker_model=ranker_model,
            scaling_stats=scaling_stats
        )
        print("Two-Stage Recommender with pure Python PyTorch engine loaded successfully.")

        # Default fallback
        ml_models["recommender"] = ml_models["recommender_native"]

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
def recommend_issues(query: str, k: int = 5, engine: str = "native"):
    """
    Accepts a search query and returns the top k recommended GitHub issues
    scored using the native C++ ranking engine (unquantized/quantized) or pure Python engine.
    """
    recommender = None
    if engine == "native":
        recommender = ml_models.get("recommender_native")
    elif engine == "quantized":
        recommender = ml_models.get("recommender_quantized")
    elif engine == "python":
        recommender = ml_models.get("recommender_python")

    # Fallback to default native if the requested engine is not initialized
    if not recommender:
        recommender = ml_models.get("recommender")
        
    if not recommender:
        return {"status": "error", "message": f"Recommender engine '{engine}' is not initialized."}
        
    recommendations = recommender.recommend(
        query=query,
        k_retrieval=settings.K_RETRIEVAL_DEFAULT,
        k_recommendations=k
    )
    
    # Identify which recommender was actually used
    if recommender == ml_models.get("recommender_python"):
        actual_engine = "python"
    elif recommender == ml_models.get("recommender_quantized"):
        actual_engine = "quantized"
    else:
        actual_engine = "native"
        
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
        "status": "success",
        "engine": actual_engine
    }
