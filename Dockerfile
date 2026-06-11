# 1. Select a lightweight, production-grade Python base image
FROM python:3.13-slim

# 2. Set environment variables for Python and Poetry
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# 3. Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml poetry.lock ./

# 6. Install production dependencies only
RUN poetry install --only main --no-root

# 7. Copy the application source code and model artifacts
COPY src/ ./src/
COPY app.py ./
COPY pipeline.joblib ./

EXPOSE 8000

# 8. Start the FastAPI gateway via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]