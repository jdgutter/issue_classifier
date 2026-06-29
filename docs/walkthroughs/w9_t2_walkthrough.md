# Walkthrough - Week 9 Task 2: Native C++ Execution Engine

We have successfully implemented the standalone native **C++ Execution Engine** (`inference_engine.cpp`) utilizing the ONNX Runtime C++ API to run model inference locally.

## Changes Made

### Native C++ Inference Engine
* **[inference_engine.cpp](file:///Users/jgutter/Documents/coding/ml_projects/github_issue_classifier/src/native/inference_engine.cpp)**: Implemented the native runner script:
  * Initializes the global ONNX Runtime environment (`Ort::Env`) and configures graph optimizations.
  * Loads `ranking_model.onnx` into memory (`Ort::Session`).
  * Wraps continuous metadata, categorical tags, and dense text embeddings into native `Ort::Value` tensors using zero-copy raw pointer mapping.
  * Runs model forward tensor execution (`session.Run`) and extracts raw outputs.
  * Implements a CLI wrapper to calculate and display the sigmoid-normalized personalization probability score.
* **[CMakeLists.txt](file:///Users/jgutter/Documents/coding/ml_projects/github_issue_classifier/src/native/CMakeLists.txt)**: Configures a portable C++ compilation system (requiring C++17), searching standard system directories for ONNX Runtime headers and library binaries.

---

## Industry Interview Context: Design & Implementation Decisions

When detailing this step in a professional or design interview setting, keep these points in focus:

### 1. Direct Memory Mapping (Zero-Copy Tensors)
* **Interview Explanation:** *"To minimize latency on hot execution paths, we wrap the raw pointer buffers of our standard `std::vector` objects directly into `Ort::Value` tensors using `Ort::Value::CreateTensor`. ONNX Runtime references the underlying memory block in-place. This avoids copying memory buffers during tensor allocation, keeping memory allocation overhead to zero and maximizing CPU cache locality."*

### 2. Standalone C++ Compilation via Clang++
* **Interview Explanation:** *"While CMake is used for cross-platform target configuration in CI/CD, we utilize a direct `clang++` command locally to build the binary against Homebrew's installed `onnxruntime` libraries (`-I/opt/homebrew/include/onnxruntime` and `-L/opt/homebrew/lib`). This proves that the core native engine compiled binary is extremely portable and has no runtime dependencies on Python or PyTorch libraries."*

---

## Compilation and Execution Verification

1. **Compilation Command**:
   ```bash
   clang++ -std=c++17 -O3 \
     -I/opt/homebrew/include/onnxruntime \
     -L/opt/homebrew/lib \
     -lonnxruntime \
     src/native/inference_engine.cpp -o src/native/inference_engine
   ```

2. **Execution Command**:
   ```bash
   ./src/native/inference_engine models/ranking_model.onnx
   ```

3. **Runtime Execution Output**:
   ```
   Initializing ONNX Runtime C++ Environment...
   Loading ONNX model from: models/ranking_model.onnx...
   Executing forward native tensor pass (Inference)...
   Inference completed successfully!
   Output Raw Logits: [0.259369]
   Sigmoid Probability Score: 0.564481
   ```

---

## Test Verification Results

We verified that the full test suite remains green, confirming all 24 tests pass successfully:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.1.1, pluggy-1.6.0 -- /Users/jgutter/Documents/coding/ml_projects/github_issue_classifier/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/jgutter/Documents/coding/ml_projects/github_issue_classifier
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 24 items

tests/test_app.py::test_predict_healthy_payload PASSED                   [  4%]
tests/test_app.py::test_predict_malformed_payload PASSED                 [  8%]
tests/test_embeddings.py::test_embedder_initialization PASSED            [ 12%]
tests/test_embeddings.py::test_embed_text_returns_correct_shape PASSED   [ 16%]
tests/test_embeddings.py::test_embed_issues_populates_embedding_field PASSED [ 20%]
tests/test_embeddings.py::test_embed_empty_lists PASSED                  [ 25%]
tests/test_ingestion.py::test_read_github_issues_csv PASSED              [ 29%]
tests/test_onnx.py::test_onnx_export_flow PASSED                         [ 33%]
tests/test_ranking.py::test_ranking_dataset PASSED                       [ 37%]
tests/test_ranking.py::test_ranking_model_forward PASSED                 [ 41%]
tests/test_ranking.py::test_train_ranking_model PASSED                   [ 45%]
tests/test_recommender.py::test_recommender_e2e_pipeline PASSED          [ 50%]
tests/test_recommender.py::test_recommender_empty_candidates PASSED      [ 54%]
tests/test_retrieval.py::test_retriever_initialization PASSED            [ 58%]
tests/test_retrieval.py::test_retriever_retrieve_candidates PASSED       [ 62%]
tests/test_schema.py::test_json_document_validation_fails_on_empty_payload PASSED [ 66%]
tests/test_schema.py::test_json_document_validation_succeeds PASSED      [ 70%]
tests/test_schema.py::test_github_issue_with_signals PASSED              [ 75%]
tests/test_schema.py::test_github_issue_validation_succeeds PASSED       [ 79%]
tests/test_schema.py::test_github_issue_validation_fails_on_empty_body PASSED [ 83%]
tests/test_vector_index.py::test_vector_index_initialization PASSED      [ 87%]
tests/test_vector_index.py::test_vector_index_build_and_load PASSED      [ 91%]
tests/test_vector_index.py::test_vector_index_search PASSED              [ 95%]
tests/test_vector_index.py::test_vector_index_exceptions PASSED          [100%]

======================= 24 passed, 4 warnings in 30.22s ========================
```
