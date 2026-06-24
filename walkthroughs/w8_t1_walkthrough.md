# Walkthrough - Week 8 Task 1: Candidate Retrieval

We have successfully implemented the first stage of our two-stage recommender: **Candidate Retrieval**.

## Changes Made

### Candidate Retrieval Engine
* **[retrieval.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/src/retrieval.py)**: Implemented the `CandidateRetriever` class.
  * Injects `IssueVectorIndex` and `IssueEmbedder` at instantiation (Dependency Injection).
  * Executes query-time dense embedding generation and queries the FAISS index for the top $K$ candidates.
  * Returns the raw list of similar `GithubIssue` records, isolating candidate generation from downstream ranking complexity.

### Verification and Tests
* **[test_retrieval.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/tests/test_retrieval.py)**: Added unit tests validating:
  * Proper initialization and dependency management.
  * Retrieval constraints (verifying candidate size is capped at $K$).
  * Proper handling when requesting more candidates than exist in the dataset (graceful degradation).
  * Semantic query ordering (e.g., database queries return PostgreSQL timeout issues first).

---

## Industry Interview Context: Design & Implementation Decisions

When describing this system in a technical design review or ML engineering interview, keep these points in focus:

### 1. Separation of Concerns (Stateless Retrieval Engine)
* **Interview Explanation:** "We designed `CandidateRetriever` to accept its dependencies (`IssueVectorIndex` and `IssueEmbedder`) via its constructor rather than hardcoding them internally. This makes the class **stateless** relative to the storage layer and embedding model. It enables clean parallelization in production, makes testing simpler by allowing index mocking, and makes the retrieval logic highly maintainable as index types or models evolve."

### 2. Isolation of Retrieval and Scoring Latency
* **Interview Explanation:** "The retrieval step purposefully strips distance scores and returns pure `GithubIssue` records. In a production pipeline, retrieval scores (such as cosine similarities) are useful for hard-cutoff filtering but are insufficient for final sorting. Final personalization relies on metadata signals (clicks, repo popularity, age) which are handled by the Stage 2 heavy scoring model. Keeping this boundary clean prevents leakage of logic and preserves system decoupling."

---

## Test Verification Results

We verified the entire test suite, confirming all 16 tests pass:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.3, pluggy-1.6.0 -- /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 16 items

tests/test_embeddings.py::test_embedder_initialization PASSED            [  6%]
tests/test_embeddings.py::test_embed_text_returns_correct_shape PASSED   [ 12%]
tests/test_embeddings.py::test_embed_issues_populates_embedding_field PASSED [ 18%]
tests/test_embeddings.py::test_embed_empty_lists PASSED                  [ 25%]
tests/test_ingestion.py::test_read_github_issues_csv PASSED              [ 31%]
tests/test_retrieval.py::test_retriever_initialization PASSED            [ 37%]
tests/test_retrieval.py::test_retriever_retrieve_candidates PASSED       [ 43%]
tests/test_schema.py::test_json_document_validation_fails_on_empty_payload PASSED [ 50%]
tests/test_schema.py::test_json_document_validation_succeeds PASSED      [ 56%]
tests/test_schema.py::test_github_issue_with_signals PASSED              [ 62%]
tests/test_schema.py::test_github_issue_validation_succeeds PASSED       [ 68%]
tests/test_schema.py::test_github_issue_validation_fails_on_empty_body PASSED [ 75%]
tests/test_vector_index.py::test_vector_index_initialization PASSED      [ 81%]
tests/test_vector_index.py::test_vector_index_build_and_load PASSED      [ 87%]
tests/test_vector_index.py::test_vector_index_search PASSED              [ 93%]
tests/test_vector_index.py::test_vector_index_exceptions PASSED          [100%]

============================= 16 passed in 20.45s ==============================
```
