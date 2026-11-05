"""Unit and integration tests for modern dataset ingestion connector."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest_modern import IngestionReport, ModernDatasetIngestor


@pytest.fixture
def ingestor(tmp_path):
    """Instantiate ingestor pointing to project raw data directory."""
    candidate = Path("analytics-fc/data/raw/kaggle")
    if not candidate.exists():
        candidate = Path("data/raw/kaggle")
    output_dir = tmp_path / "bronze"
    return ModernDatasetIngestor(raw_dir=str(candidate), output_dir=str(output_dir))


def test_locate_modern_archives(ingestor):
    """Verify modern archive detection finds FC 24 and FIFA 23 archives."""
    archives = ingestor.locate_modern_archives()
    assert "fc24" in archives
    assert archives["fc24"].exists()
    assert "archive.zip" in archives["fc24"].name.lower()


def test_stream_fc24_records_schema(ingestor):
    """Verify streamed records contain required key attributes and types."""
    archives = ingestor.locate_modern_archives()
    if "fc24" not in archives:
        pytest.skip("FC 24 archive not present")

    # Read minimal subset for quick validation
    df = ingestor.stream_fc24_records(archives["fc24"])
    assert len(df) == 180021
    assert "player_id" in df.columns
    assert "fifa_version" in df.columns
    assert "overall" in df.columns
    assert "potential" in df.columns
    assert "short_name" in df.columns
    assert df["fifa_version"].dtype == "int32" or df["fifa_version"].dtype == "int64"
    assert 24 in df["fifa_version"].values


def test_run_ingestion_end_to_end(ingestor):
    """Test full execution generating verified bronze parquet file."""
    report = ingestor.run_ingestion(
        target_archive_key="fc24", output_filename="test_granular.parquet"
    )

    assert isinstance(report, IngestionReport)
    assert report.status == "SUCCESS"
    assert report.records_extracted == 180021
    assert report.distinct_players == 53111
    assert 24 in report.editions_covered
    assert report.output_path is not None
    assert Path(report.output_path).exists()

    # Verify parquet loadability
    df_loaded = pd.read_parquet(report.output_path)
    assert len(df_loaded) == 180021
    assert df_loaded["player_id"].nunique() == 53111
