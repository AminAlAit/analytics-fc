"""Export a lightweight serving-only DuckDB and model bundle under 100 MB.

Creates a self-contained zip archive for free cloud deployments (such as
Streamlit Community Cloud) without altering canonical dbt_project/analytics_fc.duckdb
or DVC-tracked data/processed/ artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import duckdb

from src.catalog_search import ALL_SKILL_COLUMNS, CORE_SEARCH_COLUMNS

logger = logging.getLogger("analytics_fc.scripts.export_serving_bundle")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DB = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
DEFAULT_GALLERY_PATH = BASE_DIR / "data" / "processed" / "superstar_gallery.json"
DEFAULT_QUANTILE_PATH = BASE_DIR / "models" / "trajectory_regressor.joblib"
DEFAULT_CONFORMAL_PATH = BASE_DIR / "models" / "conformal_predictor.joblib"
DEFAULT_BUNDLE_OUTPUT = BASE_DIR / "dist" / "analytics_fc_serving_bundle.zip"

SERVING_FCT_COLUMNS: list[str] = [
    "player_id",
    "fifa_edition_year",
    "short_name",
    "long_name",
    "age",
    "nationality",
    "club",
    "league_name",
    "positions",
    "overall",
    "potential",
    "value_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "annual_growth_velocity",
]

SERVING_STG_COLUMNS: list[str] = [*CORE_SEARCH_COLUMNS, *ALL_SKILL_COLUMNS]
MAX_GITHUB_ASSET_BYTES = 100 * 1024 * 1024


def _src_table_exists(
    con: duckdb.DuckDBPyConnection,
    db_alias: str,
    schema_name: str,
    table_name: str,
) -> bool:
    """Return True if a table exists in the attached DuckDB catalog."""
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM duckdb_tables()
        WHERE database_name = ?
          AND schema_name = ?
          AND table_name = ?;
        """,
        [db_alias, schema_name, table_name],
    ).fetchone()
    return bool(row is not None and row[0] > 0)


def export_serving_duckdb(
    source_db: Path | str = DEFAULT_SOURCE_DB,
    dest_db: Path | str | None = None,
) -> dict[str, Any]:
    """Export only the dashboard-serving tables and columns into a slim DuckDB file."""
    src_path = Path(source_db).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source DuckDB database not found at {src_path}")

    target_path = (
        Path(dest_db).resolve()
        if dest_db is not None
        else (BASE_DIR / "dist" / "analytics_fc_serving.duckdb").resolve()
    )
    if target_path == src_path:
        raise ValueError("Destination serving DuckDB path must not overwrite source DuckDB.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".duckdb", dir=str(target_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    if tmp_path.exists():
        tmp_path.unlink()

    stg_cols_sql = ", ".join(SERVING_STG_COLUMNS)
    fct_cols_sql = ", ".join(SERVING_FCT_COLUMNS)

    stg_rows = 0
    fct_rows = 0
    con = duckdb.connect(str(tmp_path), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        escaped_src = str(src_path).replace("'", "''")
        con.execute(f"ATTACH '{escaped_src}' AS src_db (READ_ONLY);")

        con.execute(
            f"""
            CREATE TABLE main.stg_fifa_granular AS
            SELECT {stg_cols_sql}
            FROM src_db.main.stg_fifa_granular;
            """
        )
        stg_count = con.execute("SELECT COUNT(*) FROM main.stg_fifa_granular;").fetchone()
        stg_rows = int(stg_count[0]) if stg_count else 0

        if _src_table_exists(con, "src_db", "main", "fct_player_seasons"):
            con.execute(
                f"""
                CREATE TABLE main.fct_player_seasons AS
                SELECT {fct_cols_sql}
                FROM src_db.main.fct_player_seasons;
                """
            )
            fct_count = con.execute("SELECT COUNT(*) FROM main.fct_player_seasons;").fetchone()
            fct_rows = int(fct_count[0]) if fct_count else 0

        con.execute("DETACH src_db;")
        con.execute("CHECKPOINT;")
    except Exception:
        con.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    con.close()

    os.replace(tmp_path, target_path)
    db_bytes = target_path.stat().st_size
    return {
        "dest_db": str(target_path),
        "size_bytes": db_bytes,
        "size_mb": round(db_bytes / (1024 * 1024), 2),
        "stg_fifa_granular_rows": stg_rows,
        "fct_player_seasons_rows": fct_rows,
    }


def create_serving_bundle(
    source_db: Path | str = DEFAULT_SOURCE_DB,
    gallery_path: Path | str = DEFAULT_GALLERY_PATH,
    quantile_model_path: Path | str = DEFAULT_QUANTILE_PATH,
    conformal_model_path: Path | str = DEFAULT_CONFORMAL_PATH,
    output_zip: Path | str = DEFAULT_BUNDLE_OUTPUT,
) -> dict[str, Any]:
    """Package slim serving DuckDB, superstar gallery, and Model 2 artifacts into a zip."""
    gallery_file = Path(gallery_path)
    if not gallery_file.exists():
        raise FileNotFoundError(f"Superstar gallery missing at {gallery_file}")

    zip_dest = Path(output_zip).resolve()
    zip_dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(zip_dest.parent)) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        slim_db_path = tmp_dir_path / "analytics_fc.duckdb"
        db_stats = export_serving_duckdb(source_db=source_db, dest_db=slim_db_path)

        fd, tmp_zip_name = tempfile.mkstemp(suffix=".zip", dir=str(zip_dest.parent))
        os.close(fd)
        tmp_zip = Path(tmp_zip_name)
        included_files: list[str] = []

        try:
            with zipfile.ZipFile(
                tmp_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as zf:
                zf.write(slim_db_path, arcname="dbt_project/analytics_fc.duckdb")
                included_files.append("dbt_project/analytics_fc.duckdb")

                zf.write(gallery_file, arcname="data/processed/superstar_gallery.json")
                included_files.append("data/processed/superstar_gallery.json")

                for model_path, arcname in [
                    (Path(quantile_model_path), "models/trajectory_regressor.joblib"),
                    (Path(conformal_model_path), "models/conformal_predictor.joblib"),
                ]:
                    if model_path.exists():
                        zf.write(model_path, arcname=arcname)
                        included_files.append(arcname)

            os.replace(tmp_zip, zip_dest)
        except Exception:
            if tmp_zip.exists():
                tmp_zip.unlink()
            raise

    bundle_bytes = zip_dest.stat().st_size
    if bundle_bytes >= MAX_GITHUB_ASSET_BYTES:
        raise ValueError(
            f"Serving bundle size {bundle_bytes} bytes exceeds 100 MB limit ({MAX_GITHUB_ASSET_BYTES})."
        )

    return {
        "bundle_path": str(zip_dest),
        "bundle_bytes": bundle_bytes,
        "bundle_mb": round(bundle_bytes / (1024 * 1024), 2),
        "included_files": included_files,
        "db_stats": db_stats,
    }


def main() -> None:
    """CLI entrypoint for exporting a slim serving bundle."""
    parser = argparse.ArgumentParser(description="Export Analytics FC slim serving bundle.")
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    parser.add_argument("--quantile-model", type=Path, default=DEFAULT_QUANTILE_PATH)
    parser.add_argument("--conformal-model", type=Path, default=DEFAULT_CONFORMAL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE_OUTPUT)
    args = parser.parse_args()

    result = create_serving_bundle(
        source_db=args.source_db,
        gallery_path=args.gallery,
        quantile_model_path=args.quantile_model,
        conformal_model_path=args.conformal_model,
        output_zip=args.output,
    )
    print(
        f"Exported serving bundle to {result['bundle_path']} "
        f"({result['bundle_mb']} MB, files={result['included_files']})."
    )


if __name__ == "__main__":
    main()
