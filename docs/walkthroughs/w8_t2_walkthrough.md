# Walkthrough - Week 8 Task 2: Candidate Ranking

We have successfully implemented the second stage of our two-stage recommender: **Candidate Ranking** using PyTorch.

## Changes Made

### Ranking Engine
* **[ranking.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/src/ranking.py)**: Implemented the dataset pipeline, neural network structure, and training loop.
  * **`IssueRankingDataset`**: Prepares features by concatenating unit-normalized text embeddings, categorical tag embeddings, and standardized continuous features. Generates binary engagement target labels dynamically based on simulated click and popularity thresholds.
  * **`IssueRankingModel`**: An embedding-augmented MLP. Projects tag categories through a low-dimensional lookup (`nn.Embedding`), concatenates them with continuous features and the 384D text embedding, and passes them through fully-connected layers with Dropout and ReLU activations.
  * **`train_ranking_model`**: Trains the network on a list of `GithubIssue` objects using standard backpropagation with Adam and BCE loss, saving a packaged checkpoint (`ranking_model.pt`) containing state dictionaries, model hyper-parameters, and continuous scaling parameters.

### Verification and Tests
* **[test_ranking.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/tests/test_ranking.py)**: Added unit tests validating:
  * Proper continuous standardization, categorical tag clipping, and shape matching inside `IssueRankingDataset`.
  * Correct forward pass logits shapes and probability normalization inside `IssueRankingModel`.
  * Model convergence and serialization/deserialization logic inside `train_ranking_model`.

---

## Industry Interview Context: Design & Implementation Decisions

When detailing this step in a professional or design interview setting, keep these points in focus:

### 1. Unified Continuous & Categorical Input Pipeline
* **Interview Explanation:** *"We leverage a hybrid feature representation. Continuous features are standard Z-score normalized to ensure clean gradients during backpropagation. Simultaneously, categorical tags are projected through a low-dimensional embedding lookup (`nn.Embedding`), capturing tag similarities. Finally, dense text embeddings from Stage 1 are appended. This creates a unified representation that allows the downstream MLP to learn complex cross-feature interactions."*

### 2. Numerical Stability via Logit-Space Calculations
* **Interview Explanation:** *"During model training, the network's `forward` method outputs raw logit values rather than probabilities. We feed these logits directly into `nn.BCEWithLogitsLoss`. By combining the Sigmoid activation and standard Binary Cross-Entropy loss mathematically under a single operation, we gain high numerical stability (via the log-sum-exp trick), avoiding overflow and underflow problems commonly seen when calculating sigmoids and logs separately."*

### 3. Model Package Self-Containment (Metadata/Scaling Stats Serialization)
* **Interview Explanation:** *"To run native C++ inference in Week 9, we must preserve feature preprocessing statistics. When saving the model checkpoint (`ranking_model.pt`), we package the continuous feature means and standard deviations along with the model weights. This ensures that the downstream inference runner can perform identical inputs normalization dynamically, removing any hardcoded data pipeline assumptions."*

---

## Test Verification Results

We verified the entire test suite, confirming all 19 tests pass:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.3, pluggy-1.6.0 -- /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 19 items

tests/test_embeddings.py::test_embedder_initialization PASSED            [  5%]
tests/test_embeddings.py::test_embed_text_returns_correct_shape PASSED   [ 10%]
tests/test_embeddings.py::test_embed_issues_populates_embedding_field PASSED [ 15%]
tests/test_embeddings.py::test_embed_empty_lists PASSED                  [ 21%]
tests/test_ingestion.py::test_read_github_issues_csv PASSED              [ 26%]
tests/test_ranking.py::test_ranking_dataset PASSED                       [ 31%]
tests/test_ranking.py::test_ranking_model_forward PASSED                 [ 36%]
tests/test_ranking.py::test_train_ranking_model PASSED                   [ 42%]
tests/test_retrieval.py::test_retriever_initialization PASSED            [ 47%]
tests/test_retrieval.py::test_retriever_retrieve_candidates PASSED       [ 52%]
tests/test_schema.py::test_json_document_validation_fails_on_empty_payload PASSED [ 57%]
tests/test_schema.py::test_json_document_validation_succeeds PASSED      [ 63%]
tests/test_schema.py::test_github_issue_with_signals PASSED              [ 68%]
tests/test_schema.py::test_github_issue_validation_succeeds PASSED       [ 73%]
tests/test_schema.py::test_github_issue_validation_fails_on_empty_body PASSED [ 78%]
tests/test_vector_index.py::test_vector_index_initialization PASSED      [ 84%]
tests/test_vector_index.py::test_vector_index_build_and_load PASSED      [ 89%]
tests/test_vector_index.py::test_vector_index_search PASSED              [ 94%]
tests/test_vector_index.py::test_vector_index_exceptions PASSED          [100%]

============================= 19 passed in 23.75s ==============================
```
