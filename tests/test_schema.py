import pytest
from schema import JSONDocument, GithubIssue
from pydantic import ValidationError

def test_json_document_validation_fails_on_empty_payload():

    bad_data = {
        "id": "123",
        "metadata": {
            "source_system": "web"
        },
        "payload": {}
    }

    with pytest.raises(ValidationError) as exec_info:
        JSONDocument(**bad_data)

    assert "Payload dictionary cannot be empty" in str(exec_info.value)

def test_json_document_validation_succeeds():
    good_data = {
        "id": "123",
        "metadata": {
            "source_system": "web"
        },
        "payload": {
            "key": "value"
        }
    }

    doc = JSONDocument(**good_data)

    assert doc.id == "123"
    assert doc.metadata.source_system == "web"

def test_github_issue_validation_succeeds():
    good_data = {
        "issue_url": "https://github.com/zhangyuanwei/node-images/issues/123",
        "issue_title": "can't load the addon",
        "body": "error: /lib64/libc.so.6: version glibc_2.14 not found"
    }

    issue = GithubIssue(**good_data)
    assert issue.issue_title == "can't load the addon"

def test_github_issue_validation_fails_on_empty_body():
    with pytest.raises(ValidationError) as exc_info:
        GithubIssue(issue_url="https://github.com/test", issue_title="test", body="   ")
    assert "Body cannot be empty or just whitespace" in str(exc_info.value)