import faiss
import numpy as np
import json
import os
from typing import List, Tuple
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder

class IssueVectorIndex:
    """
    Manages a local FAISS vector index alongside serialized document metadata 
    to retrieve similar GitHub issues based on dense vector embeddings.
    """
    def __init__(self, index_path: str = None, metadata_path: str = None):
        from src.config import settings
        self.index_path = index_path or str(settings.FAISS_INDEX_PATH)
        self.metadata_path = metadata_path or str(settings.FAISS_METADATA_PATH)
        self.index = None
        self.issues: List[GithubIssue] = []

    def build(self, issues: List[GithubIssue], embedder: IssueEmbedder) -> None:
        """
        Generates/ensures embeddings for all issues, normalizes them, constructs
        a FAISS IndexFlatIP (Inner Product), adds the vectors, and saves both
        the index and the metadata mapping to disk.
        """
        if not issues:
            raise ValueError("No issues provided to build index.")

        # Ensure all issues have embeddings populated
        issues_to_embed = [iss for iss in issues if iss.embedding is None]
        if issues_to_embed:
            embedder.embed_issues(issues_to_embed)

        # Extract embeddings and convert to float32 numpy array
        vectors = np.array([iss.embedding for iss in issues], dtype="float32")
        
        # L2 normalize vectors in-place so Inner Product equals Cosine Similarity
        faiss.normalize_L2(vectors)

        # Determine dimensions (typically 384 for all-MiniLM-L6-v2)
        dimension = vectors.shape[1]

        # IndexFlatIP uses Inner Product
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self.issues = issues

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Write index to disk
        faiss.write_index(self.index, self.index_path)

        # Write metadata mapping to disk (strip embeddings to save space)
        metadata_list = []
        for iss in issues:
            dumped = iss.model_dump()
            dumped["embedding"] = None  # strip to save space
            metadata_list.append(dumped)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Loads the FAISS index and the matching metadata file from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)
        
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
            
        self.issues = [GithubIssue(**item) for item in metadata_list]

    def search(self, query: str, embedder: IssueEmbedder, k: int = 5) -> List[Tuple[GithubIssue, float]]:
        """
        Embeds the query string, normalizes it, performs an Inner Product search 
        on the FAISS index, and returns the top K matching GithubIssue objects 
        along with their similarity scores.
        """
        if self.index is None or not self.issues:
            raise RuntimeError("Index not loaded or built yet. Call load() or build() first.")

        # Embed query text (returns a list of lists)
        query_vector_list = embedder.embed_text([query])
        if not query_vector_list:
            return []

        query_vector = np.array(query_vector_list, dtype="float32")
        # Normalize query vector for cosine similarity mapping
        faiss.normalize_L2(query_vector)

        # Run query: returns (distances/scores, indices)
        scores, indices = self.index.search(query_vector, k)

        # Flatten outputs since we query 1 vector
        scores = scores[0]
        indices = indices[0]

        results = []
        for idx, score in zip(indices, scores):
            # FAISS returns -1 for empty/unmatched slots if k > index size
            if idx == -1 or idx >= len(self.issues):
                continue
            results.append((self.issues[idx], float(score)))

        return results
