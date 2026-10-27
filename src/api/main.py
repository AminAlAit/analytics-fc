"""Main FastAPI application entrypoint for Analytics FC.

Provides HTTP REST endpoints for player discovery, profile extraction,
vector similarity inference, and system health monitoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.routers.players import router as players_router
from src.catalog_search import get_catalog_stats, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analytics_fc.api")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance scouting discovery and ceiling matching API microservice.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Sub-Routers
app.include_router(players_router)


@app.get("/", tags=["Diagnostic"])
def root_endpoint() -> dict[str, Any]:
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Diagnostic"])
def health_probe() -> dict[str, Any]:
    """Health check probe verifying database connectivity and artifact availability."""
    db_status = "unavailable"
    catalog_records = 0
    gallery_status = "unavailable"
    gallery_size = 0

    # 1. Verify DuckDB Catalog Connectivity
    try:
        con = get_connection(settings.db_path)
        try:
            stats = get_catalog_stats(con=con)
            catalog_records = stats["total_rows"]
            db_status = "connected"
        finally:
            con.close()
    except Exception as exc:
        logger.error(f"Database health check failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {exc}",
        ) from exc

    # 2. Verify Superstar Gallery Artifact
    try:
        if settings.gallery_path.exists():
            with open(settings.gallery_path, "r", encoding="utf-8") as f:
                gallery = json.load(f)
                gallery_size = len(gallery)
                gallery_status = "loaded"
        else:
            gallery_status = "missing"
    except Exception as exc:
        logger.error(f"Gallery artifact load failed: {exc}")
        gallery_status = f"error: {exc}"

    return {
        "status": "healthy",
        "service": "analytics-fc-api",
        "version": settings.app_version,
        "database": {
            "status": db_status,
            "catalog_records": catalog_records,
            "path": str(settings.db_path),
        },
        "artifacts": {
            "gallery_status": gallery_status,
            "superstar_count": gallery_size,
            "path": str(settings.gallery_path),
        },
        "hardware_limits": {
            "threads": 1,
            "memory_limit": "1GB",
        },
    }
