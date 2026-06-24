import torch
import os
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder
from src.ranking import IssueRankingDataset, IssueRankingModel, train_ranking_model

def get_mock_issues():
    return [
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/1",
            issue_title="OAuth login failing",
            body="500 Internal server error",
            user_historical_clicks=80,
            repo_popularity_score=0.9,
            time_since_opened=4.0,
            issue_tags_encoded=[1, 2]
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/2",
            issue_title="Typo in docs",
            body="Readme spelling correction needed",
            user_historical_clicks=5,
            repo_popularity_score=0.1,
            time_since_opened=72.0,
            issue_tags_encoded=[0, 3]
        ),
        GithubIssue(
            issue_url="https://github.com/example/repo/issues/3",
            issue_title="Refactor query parser logic",
            body="Clean up old parsing module in core system",
            user_historical_clicks=20,
            repo_popularity_score=0.4,
            time_since_opened=24.0,
            issue_tags_encoded=[5, 6]
        )
    ]

def test_ranking_dataset():
    issues = get_mock_issues()
    embedder = IssueEmbedder()
    
    dataset = IssueRankingDataset(issues, embedder)
    assert len(dataset) == 3
    
    # Verify __getitem__ structure
    cont, cat, emb, label = dataset[0]
    
    # Continuous features tensor shape: (3,)
    assert cont.shape == (3,)
    assert cont.dtype == torch.float32
    
    # Categorical tag tensor shape: (2,)
    assert cat.shape == (2,)
    assert cat.dtype == torch.int64
    
    # Embedding shape: (384,)
    assert emb.shape == (384,)
    assert emb.dtype == torch.float32
    
    # Label shape: (1,) containing 0.0 or 1.0
    assert label.shape == (1,)
    assert label.item() in [0.0, 1.0]

def test_ranking_model_forward():
    model = IssueRankingModel()
    
    # Generate dummy batch (batch_size=2)
    batch_size = 2
    continuous = torch.randn(batch_size, 3)
    categorical = torch.randint(0, 10, (batch_size, 2))
    embeddings = torch.randn(batch_size, 384)
    
    # Check forward pass outputs logits
    logits = model(continuous, categorical, embeddings)
    assert logits.shape == (batch_size, 1)
    
    # Check probability prediction outputs [0.0, 1.0]
    probs = model.predict_probability(continuous, categorical, embeddings)
    assert probs.shape == (batch_size, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_train_ranking_model(tmp_path):
    issues = get_mock_issues()
    embedder = IssueEmbedder()
    model_path = str(tmp_path / "ranking_model.pt")
    
    assert not os.path.exists(model_path)
    
    # Train model for 2 epochs on mock issues
    _ = train_ranking_model(
        issues=issues,
        embedder=embedder,
        model_path=model_path,
        epochs=2,
        batch_size=2,
        lr=0.01
    )
    
    assert os.path.exists(model_path)
    
    # Load model package and check structure
    checkpoint = torch.load(model_path)
    assert "model_state_dict" in checkpoint
    assert "scaling_stats" in checkpoint
    assert "num_tags" in checkpoint
    assert checkpoint["num_tags"] == 10
    
    # Verify statistics dictionary keys
    stats = checkpoint["scaling_stats"]
    for key in ["clicks_mean", "clicks_std", "pop_mean", "pop_std", "age_mean", "age_std"]:
        assert key in stats
        assert isinstance(stats[key], float)
