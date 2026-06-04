from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from src.transform import IssueBodyExtractor

def create_model_pipeline() -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn pipeline for GitHub issue classification.
    
    Steps:
    1. extractor: Extracts the 'body' text from validated GithubIssue Pydantic objects.
    2. vectorizer: Converts the text bodies into a sparse TF-IDF feature matrix.
    3. classifier: Trains a baseline Random Forest classifier on the features.
    """
    pipeline = Pipeline(
        steps=[
            ('extractor', IssueBodyExtractor()),
            ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
    )
    return pipeline
