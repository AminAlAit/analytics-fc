"""Analytics FC data and model pipeline orchestration module.

Coordinates raw data ingestion, dbt transformations, superstar gallery
generation, and test suite verification via ephemeral Prefect flows.
"""

from __future__ import annotations
