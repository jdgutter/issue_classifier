import pytest
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder

def test_embedder_initialization():
    embedder = IssueEmbedder()
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.model is not None

def test_embed_text_returns_correct_shape():
    embedder = IssueEmbedder()
    texts = ["Fix a bug in database connection", "Add feature flag for new recommendations UI"]
    embeddings = embedder.embed_text(texts)
    
    # Check length
    assert len(embeddings) == 2
    # Check dimensions (all-MiniLM-L6-v2 is 384 dimensions)
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    # Check elements are float
    assert all(isinstance(val, float) for val in embeddings[0])

def test_embed_issues_populates_embedding_field():
    embedder = IssueEmbedder()
    issue1 = GithubIssue(
        issue_url="https://github.com/example/repo/issues/1",
        issue_title="Bug in signup page",
        body="Getting 500 error when clicking signup button"
    )
    issue2 = GithubIssue(
        issue_url="https://github.com/example/repo/issues/2",
        issue_title="Update documentation",
        body="Missing section about OAuth installation setup"
    )
    
    assert issue1.embedding is None
    assert issue2.embedding is None
    
    issues = [issue1, issue2]
    updated_issues = embedder.embed_issues(issues)
    
    # In-place check and return check
    assert len(updated_issues) == 2
    assert updated_issues[0].embedding is not None
    assert updated_issues[1].embedding is not None
    assert len(updated_issues[0].embedding) == 384
    assert len(updated_issues[1].embedding) == 384
    
    # The embeddings for these distinct issues should be different
    assert updated_issues[0].embedding != updated_issues[1].embedding

def test_embed_empty_lists():
    embedder = IssueEmbedder()
    assert embedder.embed_text([]) == []
    assert embedder.embed_issues([]) == []
