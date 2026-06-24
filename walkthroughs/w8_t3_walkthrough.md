# Walkthrough - Week 8 Task 3: System Integration & Verification

We have successfully integrated the two-stage recommendation pipeline into a unified **`TwoStageRecommender`** execution gateway.

## Changes Made

### Integrated Recommender Engine
* **[recommender.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/src/recommender.py)**: Created the `TwoStageRecommender` class:
  * Wires Stage 1 Retrieval (FAISS) and Stage 2 Ranking (PyTorch) together.
  * Encapsulates candidate retrieval, dynamic continuous feature scaling, categorical tag index formatting, and neural network inference.
  * Ensures all retrieved candidates have dense text embeddings populated before scoring.
  * Evaluates candidates under `torch.no_grad()` and `eval()` modes to maximize inference speed.
  * Sorts final outputs descending by calculated engagement probability.

### Verification and Tests
* **[test_recommender.py](file:///Users/jgutter/Documents/coding/ml_projects/gemini_ml_service/tests/test_recommender.py)**: Added unit tests validating:
  * End-to-end integration: querying `recommend()` runs FAISS search, processes outputs into PyTorch tensors, runs model forward inference, and sorts final candidates.
  * Capping outputs to `k_recommendations` and ensuring probability limits $[0.0, 1.0]$.
  * Graceful handling of empty retrieval results (returns empty list immediately to bypass unnecessary tensor allocation).

---

## Industry Interview Context: Design & Implementation Decisions

When detailing this step in a professional or design interview setting, keep these points in focus:

### 1. Encapsulated Gateway Pattern
* **Interview Explanation:** *"In production microservices, client layers should be decoupled from the internal stages of the ML pipeline. By encapsulating candidate generation, feature preprocessing, neural network inference, and final sorting inside a single `TwoStageRecommender` module, we keep the service API clean. If we decide to swap the Stage 1 vector search for a dual-tower neural network or rewrite Stage 2 in C++, the boundary remains unchanged."*

### 2. Evaluation Mode and Gradient Isolation
* **Interview Explanation:** *"For online inference, latency is a first-class citizen. Under PyTorch, layers like Dropout and Batch Normalization behave differently during training vs. evaluation. Setting `model.eval()` deactivates these training-only mechanisms. Furthermore, wrapping the forward pass in `torch.no_grad()` prevents PyTorch from building the autograd execution graph in memory. This eliminates tracking overhead, reduces RAM footprint, and speeds up forward inference latency by up to 2-3x."*

### 3. Dynamic Feature Normalization Alignment
* **Interview Explanation:** *"To prevent train-serve feature skew, the features fed to the ranking model must be normalized using the exact mean and variance values computed over the training set. We load these parameters alongside the model weights, ensuring that query-time inference inputs align perfectly with what the model learned during training."*

---

## Verification

To run and verify the integrated recommender test suite, execute:

```bash
poetry run pytest tests/test_recommender.py -v
```
