from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
from src.schema import GithubIssue

# Dictionary to hold our loaded models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the serialized pipeline.joblib model into memory on app startup.
    print("Loading model pipeline into memory...")
    ml_models["pipeline"] = joblib.load("pipeline.joblib")
    yield
    # Clean up resources on shutdown
    ml_models.clear()

# Initialize the FastAPI application
app = FastAPI(
    title="GitHub Issue Classifier API",
    description="An API serving layer that classifies GitHub issues using our trained Scikit-Learn pipeline.",
    version="1.0.0",
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
