# Multi-Stage Recommender & High-Performance C++ Extension Plan

This **4-Week Extension Plan** builds directly on top of your containerized FastAPI and CI/CD foundations. It shifts your project from a simple text classifier to a production-grade **Two-Stage Recommendation System Engine**, while introducing a high-performance **C++ inference layer** to align with elite machine learning discovery infrastructures.

---

## Week 7: The Data Engine (Signals & Dense Embeddings)
* **Goal:** Advance beyond basic tokenization ($TF-IDF$) by implementing deep semantic text embeddings and multi-signal feature engineering to mirror realistic user-creator interactions.
* **Core Skills:** Vector embeddings, dense feature representation, data engineering for recommendation systems.

### Weekly Tasks
- [X] **Task 1: Multi-Signal Data Augmentation**
    * Update your data pipeline to inject simulated metadata signals alongside the raw text (e.g., `user_historical_clicks`, `issue_tags_encoded`, `repo_popularity_score`, `time_since_opened`).
- [X] **Task 2: Dense Embedding Generation**
    * Integrate a lightweight pre-trained sentence transformer model (via Hugging Face `transformers` or `sentence-transformers`) to convert raw GitHub issue titles and descriptions into $N$-dimensional dense vector embeddings.
- [X] **Task 3: Vector Storage Foundation**
    * Set up a local vector index instance using an open-source library like **FAISS** or **ChromaDB**. Write a pipeline script to populate this index with your generated issue embeddings linked to their unique database IDs.

---

## Week 8: Two-Stage Recommender Architecture (Retrieval & Ranking)
* **Goal:** Architect the industry-standard recommendation paradigm used in massive discovery systems: scaling from millions of items down to a handful of hyper-personalized results using split retrieval and ranking stages.
* **Core Skills:** Candidate generation (Retrieval), heavy scoring models (Ranking), multi-stage prediction pipelines.

### Weekly Tasks
- [X] **Task 1: Stage 1 Pipeline – Candidate Retrieval**
    * Build a "Retrieval" module that takes an incoming user request or issue embedding, performs an Approximate Nearest Neighbors (ANN) search via your vector index, and rapidly prunes the total dataset down to a subset of the top 50 candidate issues.
- [X] **Task 2: Stage 2 Pipeline – Candidate Ranking**
    * Train a specialized ranking model (such as a `LightGBM` Ranker or a deep cross-network using PyTorch/Scikit-Learn) that ingests the 50 candidate items and uses the user-creator metadata signals from Week 7 to calculate a precise personalization probability score.
- [X] **Task 3: System Integration & Verification**
    * Wire Stage 1 and Stage 2 together into a single cohesive pipeline class. Verify that passing an input query triggers a sequential candidate generation and downstream ranking phase, outputting a sorted list of recommendations.

---

## Week 9: High-Performance C++ Inference Layer (The Native Bridge)
* **Goal:** Connect your hardware background with low-latency software engineering by compiling your trained model into a portable binary format and wrapping it in a C++ native runtime engine.
* **Core Skills:** C++ inference runtimes, ONNX serialization, Python/C++ interoperability via bindings.

### Weekly Tasks
- [X] **Task 1: Model Export & Serialization**
    * Convert your trained Week 8 Ranking model into an **ONNX (Open Neural Network Exchange)** format file (`.onnx`), ensuring all inputs, outputs, and layer shapes are strictly defined.
- [X] **Task 2: Native C++ Execution Engine**
    * Write a standalone C++ script (`inference_engine.cpp`) that utilizes the `ONNX Runtime C++ API`. The script should load your `.onnx` model file, accept an array/vector of feature inputs, execute the forward tensor pass natively, and return the predicted tensor.
- [X] **Task 3: Python-C++ Binding (`pybind11`)**
    * Implement a lightweight bridge using `pybind11` to compile your C++ engine into a shared library module (`.so` or `.pyd`) that can be imported directly into Python.
    * Update your FastAPI app (`main.py`) to swap out the Python ranking execution engine for this compiled native wrapper.

---

## Week 10: Model Optimization, Hardware Profiling & Evaluation
* **Goal:** Treat your newly deployed native engine as a true "Device Under Test" (DUT). Profile execution latencies, implement optimization strategies, and perform strict performance benchmarking.
* **Core Skills:** Quantization, CPU/Memory profiling, regression latency tracking, MLOps metrics verification.

### Weekly Tasks
- [ ] **Task 1: Post-Training Model Quantization**
    * Apply quantization techniques to your ONNX model to compress your network weights (e.g., converting 32-bit floating-point weights down to 8-bit integers (`INT8`)), optimizing execution speed and memory footprints.
- [ ] **Task 2: Hardware Level Profiling & Latency Tracking**
    * Treat the pipeline as a physical system bottleneck: Use system profiling utilities (like Linux `perf`, `gprof`, or `valgrind`) to analyze your C++ inference execution. Document the memory consumption, CPU utilization, and tail latencies ($p50$, $p95$, $p99$).
- [ ] **Task 3: Comprehensive Regression Benchmarking**
    * Write an automated performance testing script that simulates concurrent traffic to your FastAPI app. Measure and create a markdown table comparing the pure Python baseline performance against your optimized C++ native engine pipeline under load (tracking throughput in requests per second and average latency).

---

## Updated Target Project Deliverables
Upon completing this extended curriculum, your portfolio project will boast features that perfectly map to the high-performance demands of the YouTube Shorts engineering profile:
1. **Production-Grade Design:** A complete, industry-standard two-stage (Retrieval + Ranking) recommender framework.
2. **Dense Semantic Matching:** Deep semantic search foundations via vector embeddings and index searching instead of mere keyword lookups.
3. **Low-Latency Architecture:** A functional, compiled native C++ inference runner plugged directly into a Python API layer.
4. **Hardware Verification Rigor:** Comprehensive runtime metrics profiling, data quantization optimization, and strict latency benchmarking.