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

    @field_validator('body')
    @classmethod
    def check_body_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Body cannot be empty or just whitespace')
        return v
