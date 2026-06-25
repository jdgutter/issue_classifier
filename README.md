# GitHub Issue Classifier ML Service

A production-grade Machine Learning service that classifies GitHub issues. This project was built over a 6-week iterative plan, evolving from simple data ingestion to a fully containerized, tested, and CI/CD automated REST API.

## 🚀 Features

- **Robust Data Validation:** Strictly enforces data contracts using **Pydantic**.
- **Reproducible ML Pipeline:** Custom Scikit-Learn transformers and pipelines.
- **Experiment Tracking:** Tracks model runs and registers artifacts (`pipeline.joblib`) using **MLflow**.
- **High-Performance Serving:** Exposes a lightning-fast `POST /predict` endpoint using **FastAPI**.
- **Containerized:** Deterministic, reproducible deployments via **Docker** and **Poetry**.
- **Automated CI/CD:** Automated testing and strict **Ruff** linting enforced via **GitHub Actions**.

## 🛠️ Tech Stack

- **Python** 3.13
- **Dependency Management:** Poetry 2.4.1
- **Machine Learning:** Scikit-Learn
- **MLOps:** MLflow
- **Web Framework:** FastAPI & Uvicorn
- **Testing & Quality:** Pytest, Ruff
- **DevOps:** Docker, GitHub Actions

## 💻 Local Development Setup

### Prerequisites
- Python 3.12+
- Poetry (Recommended: v2.4.1)

### Installation
1. Clone the repository and navigate to the project root:
   ```bash
   git clone <your-repo-url>
   cd issue_classifier
   ```
2. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Run model training to generate the baseline classifier:
   ```bash
   poetry run python src/training/train.py
   ```
4. Run the FastAPI development server:
   ```bash
   poetry run uvicorn src.api.app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`. You can view the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## 🐳 Docker Deployment

You can easily build and run the application in a completely isolated container.

1. Build the image:
   ```bash
   docker build -t issue-classifier:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 issue-classifier:latest
   ```

## 📊 Experiment Tracking (MLflow)

To track experiments, logged metrics, parameters, and register model artifact versions:

1. Launch the local MLflow UI dashboard:
   ```bash
   poetry run mlflow ui
   ```
2. Open `http://127.0.0.1:5000` in your web browser to review training histories, evaluate metrics comparisons, and inspect generated files (e.g. `pipeline.joblib`).

## 🧪 Usage / API Reference

### `POST /predict`
Classifies a given GitHub issue.

**Example Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "issue_url": "https://github.com/fake/repo/issues/1",
  "issue_title": "Bug: Application crashes on login",
  "body": "Whenever I try to log in, the application throws a NullPointerException and closes."
}'
```

## 🧹 Testing & Linting
Run the integration tests and linter locally:
```bash
# Run tests
poetry run pytest -v

# Run Ruff linter
poetry run ruff check .
```
