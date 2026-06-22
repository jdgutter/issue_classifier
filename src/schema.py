from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional

class DocumentMetadata(BaseModel):
    source_system: str
    timestamp: Optional[str] = None

class JSONDocument(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    metadata: DocumentMetadata
    payload: Dict[str, Any]

    @field_validator('payload')
    def check_payload_not_empty(cls, v):
        if not v:
            raise ValueError('Payload dictionary cannot be empty')
        return v

class GithubIssue(BaseModel):
    """Schema for parsing rows from smaller.csv"""
    issue_url: str = Field(..., description="The URL of the GitHub issue")
    issue_title: str = Field(..., description="The title of the GitHub issue")
    body: str = Field(..., description="The detailed body/description of the issue")

    user_historical_clicks: int = Field(default=0, description="Simulated count of historical user clicks")
    repo_popularity_score: float = Field(default=0.0, description="Simulated repository popularity score [0.0 - 1.0]")
    time_since_opened: float = Field(default=0.0, description="Simulated hours since the issue was opened")
    issue_tags_encoded: list[int] = Field(default_factory=list, description="Encoded categorical list of issue tags")

    @field_validator('body')
    @classmethod
    def check_body_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Body cannot be empty or just whitespace')
        return v
