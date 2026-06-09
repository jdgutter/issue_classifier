# 6-Week GitHub Issue Classifier Project Plan

This structured, iterative plan is designed to build out a GitHub issue classifier over the next 6 weeks. 

Assuming an allocation of approximately **10 hours per week**, each week focuses on a specific, highly transferable Machine Learning Engineering (MLE) skill. By treating the machine learning model as your **"Device Under Test" (DUT)**, this plan leverages a hardware Design Verification background while introducing modern MLOps, deployment, and data engineering practices.

---

## Plan Overview & Weekly Breakdown

```
  Week 1: Data Ingestion & Validation (Pydantic / Pytest)
    │
    ▼
  Week 2: Baseline Modeling & Pipelines (Scikit-Learn)
    │
    ▼
  Week 3: Model Evaluation & Experiment Tracking (MLflow)
    │
    ▼
  Week 4: API Serving Layer (FastAPI Deployment)
    │
    ▼
  Week 5: Containerization & Reproducibility (Docker / Poetry)
    │
    ▼
  Week 6: CI/CD & Automated Testing (GitHub Actions / Ruff)
```

---

## Week 1: Robust Data Ingestion & Pipeline Foundations (Data Engineering)
* **Goal:** Build a robust ETL (Extract, Transform, Load) script that reads raw data, validates it, and prepares it for machine learning.
* **Transferable Skills:** Data validation, error handling, standard Python data manipulation.

### Weekly Tasks
- [X] **Task 1 (3 hrs): Data Extraction Foundation**
    * Write a data ingestion script using the built-in `csv` module or `pandas` to read `smaller.csv`.
- [X] **Task 2 (4 hrs): Runtime Validation & Error Handling**
    * Iterate over the loaded rows and pass them through your `GithubIssue` Pydantic schema. 
    * Write defensive logic to handle validation errors gracefully (e.g., log the errors and drop the bad rows, rather than letting the script crash).
- [X] **Task 3 (3 hrs): Verification & Mocking**
    * Expand `test.py` to robustly test the ingestion script. 
    * Mock a small CSV within your tests to ensure your script correctly separates valid rows from malformed ones.

---

## Week 2: Baseline Model & Scikit-Learn Pipelines (Modeling & Feature Engineering)
* **Goal:** Build a reproducible machine learning pipeline. Avoid Jupyter Notebooks for this phase; implement directly as modular Python code.
* **Transferable Skills:** Avoiding data leakage, feature engineering, pipeline construction.

### Weekly Tasks
* [x] **Task 1 (3 hrs): Custom Transformers**
    * [x] Define a custom Scikit-Learn `BaseEstimator` and `TransformerMixin` (such as a `JSONFlattener`) that takes your validated Pydantic objects and extracts the feature body string.
* [x] **Task 2 (4 hrs): Pipeline Construction**
    * [x] Create an end-to-end `sklearn.pipeline.Pipeline`. 
    * [x] Combine your custom transformer, a `TfidfVectorizer` (for text feature extraction), and a basic classifier like `LogisticRegression` or `RandomForestClassifier`.
* [x] **Task 3 (3 hrs): Functional Training Run**
    * [x] Write a straightforward execution/training script that feeds the clean, validated data from Week 1 into this pipeline and prints out the initial evaluation accuracy.

---

## Week 3: Model Evaluation & MLOps Integration (Experiment Tracking)
* **Goal:** Transition away from printing metrics directly to the terminal. Implement proper experiment tracking to log your model's performance metrics and save the resulting binaries.
* **Transferable Skills:** MLOps lifecycle, model registries, experiment tracking, quantitative metric evaluation.

### Weekly Tasks
* [x] **Task 1 (3 hrs): Data Splitting Best Practices**
    * [x] Update your training script to properly split your dataset into distinct training and testing sets using `train_test_split` to ensure unbiased evaluations.
* [x] **Task 2 (4 hrs): Experiment Tracking Instrumentation**
    * [x] Integrate `MLflow` into your training workflow. 
    * [x]Log key hyperparameters (such as the `max_features` parameter of your TF-IDF vectorizer or `n_estimators` for the Random Forest).
* [x] **Task 3 (3 hrs): Artifact Serialization**
    * [x] Log evaluation metrics (`F1-score`, `Precision`, `Recall`) and register the trained Scikit-learn pipeline as a versioned artifact (`.pkl` or `.joblib` file) using MLflow's tracking API.

---

## Week 4: API Serving Layer (Model Deployment)
* **Goal:** Expose your trained model to the outside world using an industry-standard production web framework.
* **Transferable Skills:** REST APIs, Model Serving infrastructure, web framework integration.

### Weekly Tasks
* [x] **Task 1 (4 hrs): Framework Setup & Endpoint Definition**
    * [x] Create a new `FastAPI` application within your repository. 
    * [x] Define a `POST /predict` endpoint that accepts a incoming JSON payload formatted strictly according to your established `GithubIssue` schema.
* [x] **Task 2 (3 hrs): Optimized State Management**
    * [x] Write a startup lifecycle event handler in FastAPI to load your saved model artifact from Week 3 into memory, preventing expensive disk reads on subsequent network requests.
* **Task 3 (3 hrs): E2E Prediction & Interface Testing**
    * [x] Connect the route handler to the in-memory model to return inferences. 
    * [x] Test the endpoint locally using `curl` or FastAPI's auto-generated interactive Swagger UI (`/docs`).

---

## Week 5: Containerization & Reproducibility (DevOps)
* **Goal:** Package your application ecosystem so it can execute deterministically on any host machine or cloud compute instance.
* **Transferable Skills:** Docker environments, build optimization, environment reproducibility.

### Weekly Tasks
* **Task 1 (3 hrs): Dockerfile Architecture**
    * Write a multi-stage or standard `Dockerfile` for your project. 
    * Select a lightweight, production-grade Python base image (e.g., `python:3.13-slim`).
* **Task 2 (4 hrs): Layer Optimization & Build Instructions**
    * Configure the Dockerfile layers to install Poetry, copy your `pyproject.toml` and `poetry.lock` manifests, install required runtime dependencies, copy over your model binary, and start the FastAPI gateway via `uvicorn`.
* **Task 3 (3 hrs): Image Build & Local Verification**
    * Build the image locally (`docker build -t issue-classifier:latest .`) and execute it (`docker run`). 
    * Debug any underlying file path resolutions or missing system dependencies that manifest during containment.

---

## Week 6: CI/CD & Automated Testing (Verification & Automation)
* **Goal:** Automate your testing, linting, and formatting checks to function identically to a regression test suite in hardware verification.
* **Transferable Skills:** Continuous Integration pipelines, automated integration testing, GitHub Actions workflows.

### Weekly Tasks
* **Task 1 (4 hrs): Integration Test Suite**
    * Utilize FastAPI's `TestClient` to write end-to-end integration tests inside `test.py`. 
    * Verify that passing a healthy payload to `/predict` returns a `200 OK` along with a prediction string, while malformed inputs securely trigger a `422 Validation Error`.
* **Task 2 (4 hrs): GitHub Actions Workflow Configuration**
    * Create a custom `.github/workflows/ci.yml` orchestration file. 
    * Configure GitHub Actions runners to check out code, cache and install Poetry environments, and run your `pytest` suite on every push or pull request to the `main` branch.
* **Task 3 (2 hrs): Automated Quality Gates**
    * Add an automated step to your CI regression pipeline to execute `ruff` to strictly enforce code formatting consistency and catch linting errors.

---

## Target Project Deliverables
By completing this 6-week timeline, you will have constructed a production-grade machine learning service that mirrors elite industry practices:
1. **Robust Data Pipeline:** Fully type-checked data loader utilizing Pydantic.
2. **Reproducible ML Pipeline:** Clear separation of transformation and modeling inside clean Python scripts.
3. **Audit Trail & Registry:** Tracking history of metrics and parameters via MLflow.
4. **Production-Ready Gateway:** A containerized FastAPI service capable of sub-100ms inference.
5. **Automated Regression Suite:** Complete CI pipeline ensuring no code changes degrade application health.
