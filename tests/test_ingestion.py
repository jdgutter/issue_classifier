import pytest
from src.ingestion import read_github_issues_csv

def test_read_github_issues_csv(tmp_path):
    # Create a mock CSV file using pytest's built-in tmp_path fixture
    mock_csv = tmp_path / "mock_issues.csv"
    mock_csv.write_text(
        "issue_url,issue_title,body\n"
        "https://github.com/test/issues/1,Test Issue 1,This is the body of issue 1\n"
        "https://github.com/test/issues/2,Test Issue 2,This is the body of issue 2\n",
        encoding="utf-8"
    )

    # Extract the data
    rows = list(read_github_issues_csv(str(mock_csv)))

    # Verify correct extraction
    assert len(rows) == 2
    assert rows[0]["issue_url"] == "https://github.com/test/issues/1"
    assert rows[0]["issue_title"] == "Test Issue 1"
    assert rows[0]["body"] == "This is the body of issue 1"