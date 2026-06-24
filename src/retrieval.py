from typing import List
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex

class CandidateRetriever:
    """
    Stage 1 Candidate Retrieval engine.
    Retrieves a pruned subset of candidates from the full issue space 
    using approximate nearest neighbor matching.
    """
    def __init__(self, vector_index: IssueVectorIndex, embedder: IssueEmbedder):
        """
        Dependency injection pattern: allows hot-swapping different indices 
        or embedding models without changing the retrieval client logic.
        """
        self.vector_index = vector_index
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 50) -> List[GithubIssue]:
        """
        Performs semantic search on the vector index and returns the top K issue candidates.
        Outputs raw GithubIssue objects ready for Stage 2 heavy ranking.
        """
        # Execute query-time inference and FAISS search
        results = self.vector_index.search(query, self.embedder, k=k)
        
        # Strip similarity scores; Stage 2 Ranker will compute personalized probability scores
        return [issue for issue, _ in results]
