from src.schema import GithubIssue
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex
from src.retrieval import CandidateRetriever

def test_retriever_initialization(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    embedder = IssueEmbedder()
    retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
    
    assert retriever.vector_index == vector_index
    assert retriever.embedder == embedder

def test_retriever_retrieve_candidates(tmp_path):
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
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/3",
            issue_title="Database connection timeout error",
            body="PostgreSQL connection pools are exhausted under high load periods"
        )
    ]
    
    vector_index.build(issues, embedder)
    retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
    
    # 1. Test k retrieval constraint
    candidates = retriever.retrieve("authentication issues", k=2)
    assert len(candidates) == 2
    assert all(isinstance(c, GithubIssue) for c in candidates)
    
    # First candidate should be the OAuth one
    assert candidates[0].issue_title == "OAuth authentication logic failing"
    
    # 2. Test k larger than dataset size behaves gracefully
    candidates_all = retriever.retrieve("database query", k=10)
    assert len(candidates_all) == 3
    assert candidates_all[0].issue_title == "Database connection timeout error"
