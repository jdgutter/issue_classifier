from typing import List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from src.schema import GithubIssue

class IssueBodyExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to extract the 'body' feature 
    from a list of validated GithubIssue Pydantic objects.
    """
    
    def fit(self, X: List[GithubIssue], y: Any = None) -> 'IssueBodyExtractor':
        """
        Stateless transformer: no fitting is necessary. 
        Simply returns self to accommodate the Scikit-Learn pipeline API.
        """
        return self
        
    def transform(self, X: List[GithubIssue]) -> List[str]:
        """
        Extracts the body text from each GitHub issue for downstream text vectorization.
        """
        return [issue.body for issue in X]