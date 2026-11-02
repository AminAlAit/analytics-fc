"""Multi-edition attribute drift analyzer and distribution shift pipeline.

Executes stratified sampling capped at 10,000 rows per edition under single-thread,
1GB RAM constraints. Computes Wasserstein distance, Kolmogorov-Smirnov statistics,
and Population Stability Index (PSI) across historical FIFA player editions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.monitoring.config import (
    ALL_MONITORED_NUMERICAL_FEATURES,
    DriftMonitoringConfig,
)

logger = logging.getLogger("analytics_fc.monitoring.drift_analyzer")


def normalize_edition_year(edition: int) -> int:
    """Normalize 2-digit edition years (e.g., 21 -> 2021)."""
    return 2000 + edition if edition < 100 else edition


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    ref_clean = reference.dropna()
    curr_clean = current.dropna()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference sample
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(ref_clean) + epsilon * len(ref_counts))
    curr_pct = (curr_counts + epsilon) / (len(curr_clean) + epsilon * len(curr_counts))

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_feature_drift(
    reference_series: pd.Series,
    current_series: pd.Series,
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.10,
) -> dict[str, Any]:
    """Compute statistical distribution drift metrics for a single feature."""
    ref_vals = reference_series.dropna().to_numpy(dtype=float)
    curr_vals = current_series.dropna().to_numpy(dtype=float)

    if len(ref_vals) == 0 or len(curr_vals) == 0:
        return {
            "wasserstein_distance": 0.0,
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "psi": 0.0,
            "drift_detected": False,
            "reference_mean": 0.0,
            "current_mean": 0.0,
            "reference_std": 0.0,
            "current_std": 0.0,
            "reference_count": 0,
            "current_count": 0,
        }

    # Wasserstein distance (Earth Mover's Distance)
    wd = float(wasserstein_distance(ref_vals, curr_vals))

    # Two-sample Kolmogorov-Smirnov test
    ks_res = ks_2samp(ref_vals, curr_vals)
    ks_stat = float(ks_res.statistic)
    ks_p = float(ks_res.pvalue)

    # Population Stability Index
    psi_metric = calculate_psi(pd.Series(ref_vals), pd.Series(curr_vals))

    drift_flag = bool((ks_p < ks_threshold) or (psi_metric > psi_threshold))

    return {
        "wasserstein_distance": round(wd, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_p_value": float(ks_p),
        "psi": round(psi_metric, 4),
        "drift_detected": drift_flag,
        "reference_mean": round(float(np.mean(ref_vals)), 2),
        "current_mean": round(float(np.mean(curr_vals)), 2),
        "reference_std": round(float(np.std(ref_vals)), 2),
        "current_std": round(float(np.std(curr_vals)), 2),
        "reference_count": int(len(ref_vals)),
        "current_count": int(len(curr_vals)),
    }


def load_edition_data(
    edition_year: int,
    db_path: Path | str | None = None,
    sample_size: int = 10000,
    random_state: int = 42,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load stratified edition sample from stg_fifa_granular capped at sample_size rows."""
    target_year = normalize_edition_year(edition_year)
    database_path = Path(db_path) if db_path else DriftMonitoringConfig().duckdb_path

    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {database_path}")

    req_cols = (
        list(columns) if columns else ["player_id", "positions", *ALL_MONITORED_NUMERICAL_FEATURES]
    )
    cols_sql = ", ".join(req_cols)

    con = duckdb.connect(str(database_path), read_only=True)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        query = f"""
            SELECT {cols_sql}
            FROM stg_fifa_granular
            WHERE fifa_edition_year = {target_year};
        """
        df = con.execute(query).df()
    finally:
        con.close()

    if len(df) == 0:
        logger.warning("No rows found for edition year %d", target_year)
        return df

    # Stratified sampling capped at sample_size
    if len(df) > sample_size and sample_size > 0:
        if "positions" in df.columns:
            strata = (
                df["positions"].fillna("UNKNOWN").astype(str).map(lambda p: p.split(",")[0].strip())
            )
            selected_indices: list[int] = []
            target_frac = sample_size / len(df)
            rng = np.random.RandomState(random_state)
            for _, group_idx in strata.groupby(strata).groups.items():
                k = max(1, int(round(len(group_idx) * target_frac)))
                chosen = rng.choice(group_idx, size=min(k, len(group_idx)), replace=False)
                selected_indices.extend(chosen)
            df = df.loc[selected_indices]
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state)
        else:
            df = df.sample(n=sample_size, random_state=random_state)

    return df.reset_index(drop=True)


def run_attribute_drift_pipeline(
    config: DriftMonitoringConfig | None = None,
    sample_size: int = 10000,
) -> dict[str, Any]:
    """Execute complete attribute drift pipeline comparing reference vs target edition."""
    cfg = config or DriftMonitoringConfig()

    ref_year = normalize_edition_year(cfg.reference_edition)
    target_year = normalize_edition_year(cfg.target_edition)

    logger.info(
        "Initiating attribute drift analysis: FIFA %d vs FIFA %d (sample ceiling: %d)",
        ref_year,
        target_year,
        sample_size,
    )

    df_ref = load_edition_data(
        edition_year=ref_year,
        db_path=cfg.duckdb_path,
        sample_size=sample_size,
        columns=["player_id", "positions", *cfg.monitored_features],
    )

    df_target = load_edition_data(
        edition_year=target_year,
        db_path=cfg.duckdb_path,
        sample_size=sample_size,
        columns=["player_id", "positions", *cfg.monitored_features],
    )

    feature_metrics: dict[str, dict[str, Any]] = {}
    drifted_count = 0

    for feature in cfg.monitored_features:
        if feature in df_ref.columns and feature in df_target.columns:
            metrics = calculate_feature_drift(
                reference_series=df_ref[feature],
                current_series=df_target[feature],
                ks_threshold=cfg.ks_p_value_threshold,
                psi_threshold=cfg.psi_threshold,
            )
            feature_metrics[feature] = metrics
            if metrics["drift_detected"]:
                drifted_count += 1

    total_evaluated = len(feature_metrics)
    drifted_share = round(drifted_count / total_evaluated, 4) if total_evaluated > 0 else 0.0
    dataset_drift = bool(drifted_share >= 0.25)

    return {
        "reference_edition": ref_year,
        "target_edition": target_year,
        "reference_sample_size": len(df_ref),
        "target_sample_size": len(df_target),
        "total_features_evaluated": total_evaluated,
        "drifted_features_count": drifted_count,
        "drifted_features_share": drifted_share,
        "dataset_drift_detected": dataset_drift,
        "features": feature_metrics,
    }
