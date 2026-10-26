FROM python:3.12-slim

# PYTHONPATH=/app lets `python scripts/x.py` and pytest import
# `support_triage_agent` and `scripts` from the project root (fixes guide gap B6).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps .env / .git / .venv / results out of this copy.
COPY . .

EXPOSE 8000
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "/app"]
