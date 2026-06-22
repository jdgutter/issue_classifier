import hashlib
from src.schema import GithubIssue

def augment_issue(issue: GithubIssue) -> GithubIssue:
    """Deterministically generates simulated metadata signals using the issue_url hash."""
    url_hash = int(hashlib.md5(issue.issue_url.encode('utf-8')).hexdigest(), 16)
    
    # Deterministic simulations
    issue.user_historical_clicks = url_hash % 100  # 0 to 99 clicks
    issue.repo_popularity_score = (url_hash % 1000) / 1000.0  # float between 0.0 and 1.0
    issue.time_since_opened = (url_hash % 168)  # up to 1 week (168 hours)
    
    # E.g., select 2 tag IDs deterministically
    issue.issue_tags_encoded = [url_hash % 10, (url_hash // 10) % 10]
    
    return issue
