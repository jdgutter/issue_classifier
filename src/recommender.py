import torch
from typing import List, Tuple, Dict
from src.schema import GithubIssue
from src.retrieval import CandidateRetriever
from src.ranking import IssueRankingModel

try:
    import src.native.native_inference_py as native_inference
except ImportError:
    native_inference = None

class TwoStageRecommender:
    """
    Unified entry point for the two-stage recommendation system.
    Wires Stage 1 Candidate Retrieval (FAISS) and Stage 2 Candidate Ranking (PyTorch or Native C++)
    together into a high-performance query execution pipeline.
    """
    def __init__(
        self,
        retriever: CandidateRetriever,
        ranker_model: IssueRankingModel = None,
        scaling_stats: Dict[str, float] = None,
        native_engine_path: str = None
    ):
        """
        Inject dependencies and model parameters.
        Includes continuous scaling stats (mean/std) to prevent train-serve feature skew.
        """
        self.retriever = retriever
        self.ranker_model = ranker_model
        self.scaling_stats = scaling_stats
        
        self.native_engine = None
        if native_engine_path:
            if native_inference is None:
                raise ImportError("native_inference compiled module was not found.")
            self.native_engine = native_inference.InferenceEngine(native_engine_path)
            print(f"Successfully loaded native C++ inference engine from {native_engine_path}")

    def recommend(
        self,
        query: str,
        k_retrieval: int = 50,
        k_recommendations: int = 5
    ) -> List[Tuple[GithubIssue, float]]:
        """
        Ingests a query string and runs the end-to-end recommendation pipeline:
        1. Generates text candidate list (Stage 1 vector retrieval).
        2. Normalizes numerical features, formats tags, and loads text embeddings.
        3. Scores candidates using PyTorch or compiled C++ ONNX engine inference (Stage 2 ranking).
        4. Sorts candidates by engagement probability and returns the top K.
        """
        # --- Stage 1: Candidate Retrieval ---
        candidates = self.retriever.retrieve(query, k=k_retrieval)
        if not candidates:
            return []

        # Ensure all candidates have their dense embeddings generated
        issues_to_embed = [iss for iss in candidates if iss.embedding is None]
        if issues_to_embed:
            self.retriever.embedder.embed_issues(issues_to_embed)

        # --- Feature Engineering & Normalization ---
        scaled_continuous = []
        categorical = []
        embeddings = []
        
        for iss in candidates:
            # 1. Continuous metadata signal scaling (Z-score normalization)
            clicks = float(iss.user_historical_clicks)
            pop = float(iss.repo_popularity_score)
            age = float(iss.time_since_opened)
            
            scaled_clicks = (clicks - self.scaling_stats["clicks_mean"]) / self.scaling_stats["clicks_std"]
            scaled_pop = (pop - self.scaling_stats["pop_mean"]) / self.scaling_stats["pop_std"]
            scaled_age = (age - self.scaling_stats["age_mean"]) / self.scaling_stats["age_std"]
            
            scaled_continuous.append([scaled_clicks, scaled_pop, scaled_age])
            
            # 2. Categorical tags formatting
            tags = list(iss.issue_tags_encoded)
            while len(tags) < 2:
                tags.append(0)
            tags = tags[:2]
            tags = [max(0, min(9, int(t))) for t in tags]
            categorical.append(tags)
            
            # 3. Dense embeddings extraction
            embeddings.append(iss.embedding)

        # --- Stage 2: Heavy Candidate Ranking ---
        if self.native_engine is not None:
            # Flatten lists to 1D flat vectors for C++ bindings
            flat_continuous = [val for item in scaled_continuous for val in item]
            flat_categorical = [val for item in categorical for val in item]
            flat_embeddings = [val for item in embeddings for val in item]
            
            probs = self.native_engine.predict_probabilities(
                flat_continuous,
                flat_categorical,
                flat_embeddings,
                len(candidates)
            )
        else:
            if self.ranker_model is None:
                raise ValueError("Neither native_engine nor ranker_model is initialized.")
            
            # Set PyTorch model to evaluation mode (deactivates dropout, batchnorm etc.)
            self.ranker_model.eval()
            
            # Isolate inference graph from autograd tracing to conserve memory and boost speed
            with torch.no_grad():
                cont_tensor = torch.tensor(scaled_continuous, dtype=torch.float32)
                cat_tensor = torch.tensor(categorical, dtype=torch.int64)
                emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
                # Run model forward pass to output probability predictions [0.0 - 1.0]
                probs = self.ranker_model.predict_probability(cont_tensor, cat_tensor, emb_tensor)
                probs = probs.squeeze(1).tolist()

        # --- Formatting and Sorting Outputs ---
        scored_candidates = list(zip(candidates, probs))
        
        # Sort in descending order by predicted score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:k_recommendations]
