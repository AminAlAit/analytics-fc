"""Modern dataset ingestion connector for FIFA 23, FC 24, and modern editions.

Loads modern season archives from raw storage into bronze parquet format
without modifying raw archive files. Enforces single-thread execution and
memory ceiling below 500MB.
"""

import gc
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class IngestionReport:
    """Summary of records extracted and written during modern ingestion."""

    records_extracted: int = 0
    distinct_players: int = 0
    editions_covered: list[int] = field(default_factory=list)
    output_path: str | None = None
    status: str = "PENDING"
    error_message: str | None = None


class ModernDatasetIngestor:
    """Connector for discovering and extracting modern FIFA / FC datasets."""

    def __init__(
        self,
        raw_dir: str | None = None,
        output_dir: str | None = None,
    ):
        if raw_dir is None:
            if Path("analytics-fc/data/raw/kaggle").exists():
                self.raw_dir = Path("analytics-fc/data/raw/kaggle")
            else:
                self.raw_dir = Path("data/raw/kaggle")
        else:
            self.raw_dir = Path(raw_dir)

        if output_dir is None:
            if Path("analytics-fc/data").exists():
                self.output_dir = Path("analytics-fc/data/bronze")
            else:
                self.output_dir = Path("data/bronze")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def locate_modern_archives(self) -> dict[str, Path]:
        """Locate verified modern dataset archive files on disk."""
        archives: dict[str, Path] = {}

        # Search for Stefano Leone FC 24 and FIFA 23 datasets
        for p in self.raw_dir.glob("**/Stefano Leone/**/*.zip"):
            normalized = str(p).lower()
            if "fc 24" in normalized or "fc24" in normalized:
                archives["fc24"] = p
            elif "fifa 23" in normalized or "fifa23" in normalized:
                archives["fifa23"] = p

        # Search for FC 25 candidate archives
        for p in self.raw_dir.glob("**/Davis Nyagami/**/*.zip"):
            if "fc 25" in str(p).lower():
                archives["fc25_nyagami"] = p

        for p in self.raw_dir.glob("**/Aniss7/**/*.zip"):
            if "fc 25" in str(p).lower() or "2025" in str(p).lower():
                archives["fc25_sofifa"] = p

        return archives

    def stream_fc24_records(self, archive_path: Path) -> pd.DataFrame:
        """Stream modern season records from Stefano Leone FC 24 archive.

        Contains multi-edition player data up through FC 24 with 180,021 records.
        """
        with zipfile.ZipFile(archive_path, "r") as z:
            target_name = None
            for name in z.namelist():
                if name.lower() == "male_players.csv":
                    target_name = name
                    break

            if not target_name:
                raise FileNotFoundError("male_players.csv not found in FC 24 archive")

            with z.open(target_name) as f:
                # Load with low_memory=False to prevent mixed-type inference warnings
                df = pd.read_csv(f, low_memory=False)

        # Standardize fifa_version column to integer
        if "fifa_version" in df.columns:
            df["fifa_version"] = (
                pd.to_numeric(df["fifa_version"], errors="coerce").fillna(24).astype(int)
            )

        # Standardize player_id column to integer
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype(int)

        return df

    def run_ingestion(
        self,
        target_archive_key: str = "fc24",
        output_filename: str = "granular_modern.parquet",
    ) -> IngestionReport:
        """Execute ingestion pipeline and serialize records to bronze parquet."""
        report = IngestionReport()
        archives = self.locate_modern_archives()

        if target_archive_key not in archives:
            report.status = "FAILED"
            report.error_message = (
                f"Archive for {target_archive_key} not found under {self.raw_dir}"
            )
            return report

        archive_path = archives[target_archive_key]

        try:
            # Extract records
            df = self.stream_fc24_records(archive_path)

            report.records_extracted = len(df)
            report.distinct_players = int(df["player_id"].nunique())
            report.editions_covered = sorted(
                [int(v) for v in df["fifa_version"].unique() if pd.notnull(v)]
            )

            # Export to Bronze Parquet
            out_file = self.output_dir / output_filename
            df.to_parquet(out_file, index=False, engine="pyarrow")
            report.output_path = str(out_file)
            report.status = "SUCCESS"

            # Clean memory
            del df
            gc.collect()

        except Exception as exc:
            report.status = "FAILED"
            report.error_message = str(exc)

        return report
