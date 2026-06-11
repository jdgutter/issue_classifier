from fastapi.testclient import TestClient
from app import app

def test_predict_healthy_payload():
    """Verify that a well-formed payload returns a 200 OK and a valid prediction."""
    healthy_payload = {
        "issue_title": "Bug: Application crashes on login",
        "body": "Whenever I try to log in, the application throws a NullPointerException and closes.",
        "issue_url": "https://github.com/fake/repo/issues/1"
    }
    
    # Using TestClient in a 'with' block triggers the app's lifespan handler 
    # so the ML model actually gets loaded into memory.
    with TestClient(app) as client:
        response = client.post("/predict", json=healthy_payload)
        
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert data["status"] == "success"
    assert data["issue_title"] == healthy_payload["issue_title"]
    assert "predicted_label" in data

def test_predict_malformed_payload():
    """Verify that a malformed payload is caught by Pydantic and returns a 422 Validation Error."""
    malformed_payload = {
        # Missing the required 'issue_title'
        "issue_body": "This is an issue body but it lacks a title."
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=malformed_payload)
        
    assert response.status_code == 422
    assert "detail" in response.json()
