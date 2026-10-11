FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core terminal codebase
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY dbt_project/ ./dbt_project/
COPY data/processed/ ./data/processed/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py"]
