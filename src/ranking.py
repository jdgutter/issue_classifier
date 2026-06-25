import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List, Dict, Tuple
from src.schema import GithubIssue
from src.embeddings import IssueEmbedder

class IssueRankingDataset(Dataset):
    """
    Prepares GitHub issue features (continuous, categorical, dense text embeddings)
    and heuristic engagement labels for PyTorch training.
    """
    def __init__(self, issues: List[GithubIssue], embedder: IssueEmbedder, scaling_stats: Dict[str, float] = None):
        self.issues = issues
        
        # Ensure embeddings are populated
        issues_to_embed = [iss for iss in issues if iss.embedding is None]
        if issues_to_embed:
            embedder.embed_issues(issues_to_embed)
            
        # Collect raw numerical columns
        self.clicks = np.array([iss.user_historical_clicks for iss in issues], dtype=np.float32)
        self.popularity = np.array([iss.repo_popularity_score for iss in issues], dtype=np.float32)
        self.age = np.array([iss.time_since_opened for iss in issues], dtype=np.float32)
        
        # Calculate or load scaling parameters (Z-score normalization)
        if scaling_stats is None:
            self.clicks_mean, self.clicks_std = np.mean(self.clicks), np.std(self.clicks) or 1.0
            self.pop_mean, self.pop_std = np.mean(self.popularity), np.std(self.popularity) or 1.0
            self.age_mean, self.age_std = np.mean(self.age), np.std(self.age) or 1.0
        else:
            self.clicks_mean = scaling_stats["clicks_mean"]
            self.clicks_std = scaling_stats["clicks_std"]
            self.pop_mean = scaling_stats["pop_mean"]
            self.pop_std = scaling_stats["pop_std"]
            self.age_mean = scaling_stats["age_mean"]
            self.age_std = scaling_stats["age_std"]

        # Scale continuous features
        self.scaled_clicks = (self.clicks - self.clicks_mean) / self.clicks_std
        self.scaled_popularity = (self.popularity - self.pop_mean) / self.pop_std
        self.scaled_age = (self.age - self.age_mean) / self.age_std
        
        # Parse categorical issue tags (ensure length is 2 and values fit in [0, 9])
        self.tag_features = []
        for iss in issues:
            tags = list(iss.issue_tags_encoded)
            while len(tags) < 2:
                tags.append(0)
            tags = tags[:2]
            # Clip index bounds defensively
            tags = [max(0, min(9, int(t))) for t in tags]
            self.tag_features.append(tags)
        self.tag_features = np.array(self.tag_features, dtype=np.int64)
        
        # Extract text embeddings
        self.embeddings = np.array([iss.embedding for iss in issues], dtype=np.float32)
        
        # Heuristic target label generation:
        # Engaged (1) if click rates and repository popularity scores are relatively high,
        # adjusted downward if the issue has been open for too long.
        self.labels = []
        for clicks, pop, age in zip(self.clicks, self.popularity, self.age):
            score = (clicks / 100.0) * 0.5 + pop * 0.4 - (age / 168.0) * 0.2
            self.labels.append(1.0 if score > 0.3 else 0.0)
        self.labels = np.array(self.labels, dtype=np.float32)

    def get_scaling_stats(self) -> Dict[str, float]:
        return {
            "clicks_mean": float(self.clicks_mean),
            "clicks_std": float(self.clicks_std),
            "pop_mean": float(self.pop_mean),
            "pop_std": float(self.pop_std),
            "age_mean": float(self.age_mean),
            "age_std": float(self.age_std)
        }

    def __len__(self) -> int:
        return len(self.issues)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous = torch.tensor([
            self.scaled_clicks[idx],
            self.scaled_popularity[idx],
            self.scaled_age[idx]
        ], dtype=torch.float32)
        
        categorical = torch.tensor(self.tag_features[idx], dtype=torch.int64)
        text_emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return continuous, categorical, text_emb, label


class IssueRankingModel(nn.Module):
    """
    Deep candidate ranking model mapping sparse categorical tags, standardized
    continuous metadata, and dense text embeddings into a personalized click probability.
    """
    def __init__(self, num_tags: int = None, tag_embed_dim: int = None, embedding_dim: int = None):
        super().__init__()
        from src.config import settings
        num_tags = num_tags or settings.NUM_CATEGORICAL_TAGS
        tag_embed_dim = tag_embed_dim or settings.CATEGORICAL_TAG_EMBED_DIM
        embedding_dim = embedding_dim or settings.EMBEDDING_DIMENSION
        
        self.tag_embeddings = nn.Embedding(num_tags, tag_embed_dim)
        
        # Dimensions: 3 (continuous) + 2 * tag_embed_dim (categorical embeddings) + 384 (text embedding)
        mlp_input_dim = 3 + (2 * tag_embed_dim) + embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, continuous_features: torch.Tensor, categorical_features: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Executes the network forward pass, outputting raw logit scores.
        We return logits directly to ensure numerical stability during training 
        with BCEWithLogitsLoss.
        """
        # Embed categorical tags: (batch_size, 2) -> (batch_size, 2, tag_embed_dim)
        tag_embeds = self.tag_embeddings(categorical_features)
        # Flatten embeddings: (batch_size, 2 * tag_embed_dim)
        tag_embeds_flat = tag_embeds.view(tag_embeds.size(0), -1)
        
        # Concatenate continuous, categorical, and text embedding features
        x = torch.cat([continuous_features, tag_embeds_flat, text_embeddings], dim=1)
        
        return self.mlp(x)

    def predict_probability(self, continuous_features: torch.Tensor, categorical_features: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Runs inference to output standard probabilities [0.0, 1.0] using a Sigmoid activation."""
        logits = self.forward(continuous_features, categorical_features, text_embeddings)
        return torch.sigmoid(logits)


def train_ranking_model(
    issues: List[GithubIssue],
    embedder: IssueEmbedder,
    model_path: str = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 0.001
) -> IssueRankingModel:
    """Trains the ranking network using BCEWithLogitsLoss and saves the artifact to disk."""
    from src.config import settings
    model_path = model_path or str(settings.RANKER_MODEL_PATH)
    dataset = IssueRankingDataset(issues, embedder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = IssueRankingModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for cont, cat, emb, label in dataloader:
            optimizer.zero_grad()
            logits = model(cont, cat, emb)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * cont.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    # Packaging the artifact for production loading
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "scaling_stats": dataset.get_scaling_stats(),
        "num_tags": settings.NUM_CATEGORICAL_TAGS,
        "tag_embed_dim": settings.CATEGORICAL_TAG_EMBED_DIM,
        "embedding_dim": settings.EMBEDDING_DIMENSION
    }
    torch.save(state, model_path)
    print(f"Saved ranking model to {model_path}")
    return model
