# Walkthrough - Week 7 Task 3: Vector Storage Foundation

We have successfully implemented the vector storage and retrieval index using **FAISS** (`faiss-cpu`) to support fast semantic candidate retrieval.

## Changes Made

### Vector Storage Engine
* **[vector_index.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/src/vector_index.py)**: Implemented the `IssueVectorIndex` class:
  * **Build**: Validates and normalizes embeddings (L2 normalized in-place) so that fast Inner Product searches compute cosine similarities. Adds the vectors to `faiss.IndexFlatIP`.
  * **Decoupled Metadata Storage**: Serializes the list of `GithubIssue` objects (without the large embedding vectors) to a JSON file (`issues_metadata.json`) to keep disk space minimal.
  * **Search**: Embeds query, normalizes it, retrieves the top $K$ closest FAISS indices, maps them back using the metadata lookup file, and returns the sorted issues and their scores.

### Verification and Tests
* **[test_vector_index.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/tests/test_vector_index.py)**: Added unit tests validating:
  * Proper file creation on index builds.
  * Correct persistence and restoration when calling `load()`.
  * Semantic correctness: queries like "Google login error" match the authentication issue with a higher score than the documentation issue.
  * Standard MLOps edge cases (empty issue lists, non-existent files, and uninitialized search calls).

---

## Test Verification Results

We verified the index using the pytest suite:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.3, pluggy-1.6.0 -- /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/jgutter/Documents/coding/ml_projects/gemini_ml_service
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 14 items

tests/test_embeddings.py::test_embedder_initialization PASSED            [  7%]
tests/test_embeddings.py::test_embed_text_returns_correct_shape PASSED   [ 14%]
tests/test_embeddings.py::test_embed_issues_populates_embedding_field PASSED [ 21%]
tests/test_embeddings.py::test_embed_empty_lists PASSED                  [ 28%]
tests/test_ingestion.py::test_read_github_issues_csv PASSED              [ 35%]
tests/test_schema.py::test_json_document_validation_fails_on_empty_payload PASSED [ 42%]
tests/test_schema.py::test_json_document_validation_succeeds PASSED      [ 50%]
tests/test_schema.py::test_github_issue_with_signals PASSED              [ 57%]
tests/test_schema.py::test_github_issue_validation_succeeds PASSED       [ 64%]
tests/test_schema.py::test_github_issue_validation_fails_on_empty_body PASSED [ 71%]
tests/test_vector_index.py::test_vector_index_initialization PASSED      [ 78%]
tests/test_vector_index.py::test_vector_index_build_and_load PASSED      [ 85%]
tests/test_vector_index.py::test_vector_index_search PASSED              [ 92%]
tests/test_vector_index.py::test_vector_index_exceptions PASSED          [100%]

============================= 14 passed in 17.00s ==============================
```
