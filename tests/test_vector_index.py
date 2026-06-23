import pytest
import os
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex

def test_vector_index_initialization(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    assert vector_index.index_path == index_path
    assert vector_index.metadata_path == metadata_path
    assert vector_index.index is None
    assert len(vector_index.issues) == 0

def test_vector_index_build_and_load(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    embedder = IssueEmbedder()
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    
    issues = [
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/1",
            issue_title="OAuth authentication logic failing",
            body="Users are seeing 500 error when clicking the sign-in with Google button"
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/2",
            issue_title="Documentation typo in landing page",
            body="Fix a small spelling mistake in the landing page README file"
        )
    ]
    
    # Verify building creates the files
    assert not os.path.exists(index_path)
    assert not os.path.exists(metadata_path)
    
    vector_index.build(issues, embedder)
    
    assert os.path.exists(index_path)
    assert os.path.exists(metadata_path)
    
    # Load into a new index instance
    new_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    new_index.load()
    
    assert new_index.index is not None
    assert len(new_index.issues) == 2
    assert new_index.issues[0].issue_title == "OAuth authentication logic failing"
    assert new_index.issues[1].issue_title == "Documentation typo in landing page"

def test_vector_index_search(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    embedder = IssueEmbedder()
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    
    issues = [
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/1",
            issue_title="OAuth authentication logic failing",
            body="Users are seeing 500 error when clicking the sign-in with Google button"
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/2",
            issue_title="Documentation typo in landing page",
            body="Fix a small spelling mistake in the landing page README file"
        )
    ]
    
    vector_index.build(issues, embedder)
    
    # Query for login issues should return the OAuth issue first
    results_login = vector_index.search(query="Google login error", embedder=embedder, k=2)
    assert len(results_login) == 2
    
    first_issue, first_score = results_login[0]
    second_issue, second_score = results_login[1]
    
    assert first_issue.issue_title == "OAuth authentication logic failing"
    assert second_issue.issue_title == "Documentation typo in landing page"
    assert first_score > second_score  # The OAuth issue should be closer semantically
    
    # Query for doc changes should return the typo issue first
    results_typo = vector_index.search(query="fix spelling typo in docs readme", embedder=embedder, k=2)
    assert len(results_typo) == 2
    assert results_typo[0][0].issue_title == "Documentation typo in landing page"
    assert results_typo[0][1] > results_typo[1][1]

def test_vector_index_exceptions(tmp_path):
    index_path = str(tmp_path / "non_existent.faiss")
    metadata_path = str(tmp_path / "non_existent.json")
    
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    
    # Loading non-existent path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        vector_index.load()
        
    # Searching before loading/building raises RuntimeError
    embedder = IssueEmbedder()
    with pytest.raises(RuntimeError):
        vector_index.search("query", embedder)
        
    # Building with empty issue list raises ValueError
    with pytest.raises(ValueError):
        vector_index.build([], embedder)
