from src.schema import GithubIssue
from src.embeddings import IssueEmbedder
from src.vector_index import IssueVectorIndex
from src.retrieval import CandidateRetriever
from src.ranking import IssueRankingModel
from src.recommender import TwoStageRecommender

def test_recommender_e2e_pipeline(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    # 1. Setup mock issues
    issues = [
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/1",
            issue_title="Database deadlock in Postgres pool",
            body="Exceeded pool size causing query connection timeout",
            user_historical_clicks=95,
            repo_popularity_score=0.9,
            time_since_opened=2.0,
            issue_tags_encoded=[1, 2]
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/2",
            issue_title="Documentation grammar typos",
            body="Fix double spacing in installation guides and landing page readme",
            user_historical_clicks=5,
            repo_popularity_score=0.1,
            time_since_opened=120.0,
            issue_tags_encoded=[0, 3]
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/3",
            issue_title="Authentication tokens expired redirect fails",
            body="User receives infinite redirection loop when OAuth token is expired",
            user_historical_clicks=50,
            repo_popularity_score=0.6,
            time_since_opened=24.0,
            issue_tags_encoded=[5, 6]
        )
    ]
    
    # 2. Build mock FAISS vector storage index
    embedder = IssueEmbedder()
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    vector_index.build(issues, embedder)
    
    # 3. Instantiate retrieval and ranking stages
    retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
    ranker = IssueRankingModel()
    
    # Pre-calculated scaling statistics representing mock dataset distribution
    scaling_stats = {
        "clicks_mean": 50.0,
        "clicks_std": 20.0,
        "pop_mean": 0.5,
        "pop_std": 0.2,
        "age_mean": 48.0,
        "age_std": 30.0
    }
    
    # 4. Integrate stages into TwoStageRecommender pipeline
    recommender = TwoStageRecommender(
        retriever=retriever,
        ranker_model=ranker,
        scaling_stats=scaling_stats
    )
    
    # 5. Query for recommendations
    recommendations = recommender.recommend(
        query="database deadlock error",
        k_retrieval=3,
        k_recommendations=2
    )
    
    # 6. Perform validation
    assert len(recommendations) == 2
    
    # Check return type structure
    first_hit, first_score = recommendations[0]
    second_hit, second_score = recommendations[1]
    
    assert isinstance(first_hit, GithubIssue)
    assert isinstance(second_hit, GithubIssue)
    assert isinstance(first_score, float)
    assert isinstance(second_score, float)
    
    # Scores must be sorted in descending order
    assert first_score >= second_score
    
    # Ensure scores are normalized engagement probabilities [0.0 - 1.0]
    assert 0.0 <= first_score <= 1.0
    assert 0.0 <= second_score <= 1.0

def test_recommender_empty_candidates(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")
    
    embedder = IssueEmbedder()
    vector_index = IssueVectorIndex(index_path=index_path, metadata_path=metadata_path)
    # Empty index build will raise error, so we verify with unpopulated retriever returning empty list
    retriever = CandidateRetriever(vector_index=vector_index, embedder=embedder)
    ranker = IssueRankingModel()
    scaling_stats = {
        "clicks_mean": 50.0, "clicks_std": 20.0,
        "pop_mean": 0.5, "pop_std": 0.2,
        "age_mean": 48.0, "age_std": 30.0
    }
    
    recommender = TwoStageRecommender(
        retriever=retriever,
        ranker_model=ranker,
        scaling_stats=scaling_stats
    )
    
    # Mock retriever retrieve method to return empty list
    def mock_retrieve(query, k):
        return []
    retriever.retrieve = mock_retrieve
    
    # Pipeline should immediately exit and return empty list
    results = recommender.recommend("empty queries", k_retrieval=5, k_recommendations=2)
    assert results == []
