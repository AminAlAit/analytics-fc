"""Unit tests for Evidently AI configuration, feature groupings, and framework loading."""

from pathlib import Path

import evidently
from evidently import Report
from evidently.presets import DataDriftPreset

from scripts.ceiling_matcher import SKILL_COLS
from src.monitoring.config import (
    ALL_MONITORED_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    HEADLINE_NUMERICAL_FEATURES,
    TACTICAL_SKILL_FEATURES,
    DriftMonitoringConfig,
)


def test_drift_monitoring_config_defaults():
    """Verify default drift monitoring parameters and canonical paths."""
    config = DriftMonitoringConfig()

    assert config.reference_edition == 21
    assert config.target_edition == 22
    assert config.ks_p_value_threshold == 0.05
    assert config.psi_threshold == 0.10

    assert isinstance(config.duckdb_path, Path)
    assert config.duckdb_path.name == "analytics_fc.duckdb"
    assert isinstance(config.report_output_dir, Path)
    assert config.report_output_dir.name == "drift"


def test_feature_grouping_membership_and_integrity():
    """Verify feature groupings cover all 34 tactical skills without duplicates."""
    assert len(TACTICAL_SKILL_FEATURES) == 34
    assert set(TACTICAL_SKILL_FEATURES) == set(SKILL_COLS)

    # Ensure no duplicates in numerical or categorical groups
    assert len(ALL_MONITORED_NUMERICAL_FEATURES) == len(set(ALL_MONITORED_NUMERICAL_FEATURES))
    assert len(CATEGORICAL_FEATURES) == len(set(CATEGORICAL_FEATURES))

    # Verify key attributes are included
    for col in ["overall", "potential", "value_eur", "wage_eur", "age"]:
        assert col in HEADLINE_NUMERICAL_FEATURES

    for col in ["nationality", "club", "league_name", "preferred_foot"]:
        assert col in CATEGORICAL_FEATURES


def test_evidently_framework_loading():
    """Verify Evidently AI library and report presets instantiate cleanly."""
    assert evidently.__version__ != ""
    report = Report(metrics=[DataDriftPreset()])
    assert report is not None
