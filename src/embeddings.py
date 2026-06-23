from sentence_transformers import SentenceTransformer
from typing import List
from src.schema import GithubIssue

class IssueEmbedder:
    """
    A class to handle the dense embedding generation for GitHub issues
    using the sentence-transformers library.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # SentenceTransformer loads the pre-trained weights
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generates N-dimensional dense vector embeddings for a list of raw texts.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_issues(self, issues: List[GithubIssue]) -> List[GithubIssue]:
        """
        Generates embeddings for a list of GitHub issues using their title and body.
        Modifies the issues in-place by setting their 'embedding' field, and returns the list.
        """
        if not issues:
            return []
        
        # Combine issue title and body to capture both summary and detailed content
        texts = [f"Title: {issue.issue_title}\nDescription: {issue.body}" for issue in issues]
        embeddings = self.embed_text(texts)
        
        for issue, emb in zip(issues, embeddings):
            issue.embedding = emb
            
        return issues
