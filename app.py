from fastapi import FastAPI
from src.schema import GithubIssue

# Initialize the FastAPI application
app = FastAPI(
    title="GitHub Issue Classifier API",
    description="An API serving layer that classifies GitHub issues using our trained Scikit-Learn pipeline.",
    version="1.0.0"
)

@app.post("/predict")
def predict_issue(issue: GithubIssue):
    """
    Accepts a JSON payload strictly formatted to the GithubIssue schema.
    Returns a predicted classification label.
    """
    # TODO (Week 4, Task 2 & 3): 
    # 1. Load the serialized pipeline.joblib model into memory on app startup.
    # 2. Pass the validated Pydantic object to the model pipeline to get a real prediction.
    
    return {
        "issue_title": issue.issue_title,
        "predicted_label": "bug",  # Placeholder for the actual model inference
        "status": "success"
    }
