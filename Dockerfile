# Multi-stage container build for Analytics FC API and Dashboard services
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.34.0" "pydantic-settings>=2.0.0" "psutil>=5.9.0" "matplotlib>=3.9.0"

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    DUCKDB_THREADS=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser dashboard/ ./dashboard/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser dbt_project/ ./dbt_project/
COPY --chown=appuser:appuser data/processed/ ./data/processed/
COPY --chown=appuser:appuser data/monitoring/ ./data/monitoring/

RUN mkdir -p /app/models && \
    chown appuser:appuser /app /app/models

USER appuser

EXPOSE 8000 8501

CMD ["streamlit", "run", "dashboard/app.py"]
