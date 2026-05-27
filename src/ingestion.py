import csv
from typing import Iterator, Dict, Any

def read_github_issues_csv(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Reads a CSV file containing GitHub issues and yields each row as a dictionary.
    This handles the data extraction foundation (Week 1, Task 1).
    
    Design Decisions (The "Whys"):
    1. Built-in csv: Avoids heavy external dependencies like pandas, keeping the application lightweight.
    2. Iterator/yield: Provides memory efficiency by processing one row at a time rather than loading everything into RAM.
    3. Dict[str, Any] return type: Prepares data perfectly for Pydantic schema unpacking in Task 2.
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                yield row
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        raise