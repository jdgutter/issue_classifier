from fastapi.testclient import TestClient
from src.api.app import app
from unittest.mock import patch

@patch("joblib.load")
def test_predict_healthy_payload(mock_load):
    """Verify that a well-formed payload returns a 200 OK and a valid prediction."""
    healthy_payload = {
        "issue_title": "Bug: Application crashes on login",
        "body": "Whenever I try to log in, the application throws a NullPointerException and closes.",
        "issue_url": "https://github.com/fake/repo/issues/1"
    }

    # Setup dummy mock model behavior
    mock_model = mock_load.return_value
    mock_model.predict.return_value = ["bug"]
    
    # Using TestClient in a 'with' block triggers the app's lifespan handler 
    # so the ML model actually gets loaded into memory.
    with TestClient(app) as client:
        response = client.post("/predict", json=healthy_payload)
        
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert data["status"] == "success"
    assert data["issue_title"] == healthy_payload["issue_title"]
    assert "predicted_label" in data

@patch("joblib.load")
def test_predict_malformed_payload(mock_load):
    """Verify that a malformed payload is caught by Pydantic and returns a 422 Validation Error."""
    malformed_payload = {
        # Missing the required 'issue_title'
        "issue_body": "This is an issue body but it lacks a title."
    }

    # We only need to mock joblib.load because TestClient lifespan triggers on startup.
    # The model's predict method is never called because FastAPI's validation layer 
    # intercepts the malformed payload and returns 422 before invoking the route handler.
    
    with TestClient(app) as client:
        response = client.post("/predict", json=malformed_payload)
        
    assert response.status_code == 422
    assert "detail" in response.json()
