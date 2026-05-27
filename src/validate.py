from ingestion import read_github_issues_csv
from schema import GithubIssue
from pydantic import ValidationError
from typing import Iterator

def validate_github_issues(csv_file: str) -> Iterator[GithubIssue]:

    try:
        for row in read_github_issues_csv(csv_file):
            try:
                # Unpack the dictionary directly into the Pydantic model
                issue = GithubIssue(**row)
                yield issue
            except ValidationError as e:
                # Defensive logic to gracefully handle validation errors
                # We log the error and effectively drop the bad row
                print(f"Validation error for issue {row.get('issue_url', 'Unknown')}: {e}")

    except FileNotFoundError:
        print(f"Error: Unable to find file {csv_file}")

    except Exception as e:
        print(f"Unexpected error occurred {e}")
        raise
