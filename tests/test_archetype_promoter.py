"""Unit and integration tests for Automated Archetype Discovery & Promotion Pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import numpy as np
import pytest

from src.api.cache_manager import AtomicArtifactSlot
from src.api.v2.dependencies import ModelRegistry
from src.models.archetype_indexer import ArchetypeVectorIndex
from src.models.archetype_promoter import (
    ALL_SKILL_COLS,
    apply_promoter_hardware_limits,
    atomic_reload_archetype_index,
    discover_prime_candidates,
    evaluate_vector_novelty,
    export_versioned_gallery,
    run_archetype_promotion_pipeline,
)


def _sample_star(
    player_id: int,
    short_name: str,
    primary_pos: str,
    sector: str,
    peak_overall: int,
    attr_pattern: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construct a valid superstar_gallery.json entry for tests."""
    attrs = {col: (12.0 if col.startswith("goalkeeping_") else 75.0) for col in ALL_SKILL_COLS}
    if primary_pos == "GK":
        attrs = {col: (88.0 if col.startswith("goalkeeping_") else 32.0) for col in ALL_SKILL_COLS}
    if attr_pattern:
        attrs.update(attr_pattern)

    return {
        "player_id": player_id,
        "short_name": short_name,
        "long_name": short_name,
        "nationality": "Testland",
        "primary_position": primary_pos,
        "positions": primary_pos,
        "tactical_sector": sector,
        "peak_overall": peak_overall,
        "peak_seasons_count": 2,
        "peak_age_min": 25,
        "peak_age_max": 26,
        "peak_editions_observed": "2022-2023",
        "height_cm": 182.0,
        "weight_kg": 76.0,
        "attributes": attrs,
    }


def _create_synthetic_promoter_db(db_path: Path) -> None:
    """Create a minimal DuckDB with fct_player_seasons, dim_players, and stg_fifa_granular."""
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        con.execute(
            """
            CREATE TABLE main.dim_players (
                player_id INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                nationality VARCHAR,
                height_cm DOUBLE,
                weight_kg DOUBLE
            );
            """
        )
        con.execute(
            """
            INSERT INTO main.dim_players VALUES
                (9001, 'A. Striker', 'Alex Striker', 'Brazil', 184.0, 79.0),
                (9002, 'B. Modern24', 'Bruno Modern', 'France', 181.0, 75.0),
                (9003, 'C. Clone', 'Chris Clone', 'Spain', 178.0, 72.0),
                (9004, 'D. VeteranGap', 'Dan Gap', 'Italy', 185.0, 81.0),
                (9005, 'E. Keeper', 'Emil Keeper', 'Germany', 193.0, 86.0);
            """
        )

        con.execute(
            """
            CREATE TABLE main.fct_player_seasons (
                player_id INTEGER,
                fifa_edition_year INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                age INTEGER,
                nationality VARCHAR,
                positions VARCHAR,
                height_cm DOUBLE,
                weight_kg DOUBLE,
                overall INTEGER,
                pace DOUBLE,
                shooting DOUBLE,
                passing DOUBLE,
                dribbling DOUBLE,
                defending DOUBLE,
                physic DOUBLE
            );
            """
        )
        con.execute(
            """
            INSERT INTO main.fct_player_seasons VALUES
                -- 9001: Qualifying prime striker (ages 24-25, 2022-2023, OVR 89-90)
                (9001, 2022, 'A. Striker', 'Alex Striker', 24, 'Brazil', 'ST', 184.0, 79.0, 89, 91.0, 92.0, 72.0, 84.0, 42.0, 86.0),
                (9001, 2023, 'A. Striker', 'Alex Striker', 25, 'Brazil', 'ST', 184.0, 79.0, 90, 92.0, 93.0, 74.0, 85.0, 43.0, 87.0),
                -- 9002: Modern 2023-2024 prime defender with no stg_fifa_granular rows (macro fallback)
                (9002, 2023, 'B. Modern24', 'Bruno Modern', 26, 'France', 'CB', 181.0, 75.0, 88, 82.0, 52.0, 78.0, 74.0, 90.0, 88.0),
                (9002, 2024, 'B. Modern24', 'Bruno Modern', 27, 'France', 'CB', 181.0, 75.0, 89, 83.0, 54.0, 79.0, 75.0, 91.0, 89.0),
                -- 9003: Qualifying age/OVR but deliberately clones baseline star vector
                (9003, 2022, 'C. Clone', 'Chris Clone', 25, 'Spain', 'CM', 178.0, 72.0, 89, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0),
                (9003, 2023, 'C. Clone', 'Chris Clone', 26, 'Spain', 'CM', 178.0, 72.0, 89, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0),
                -- 9004: Non-consecutive seasons (2019 and 2022) must not qualify when 2+ consecutive required
                (9004, 2019, 'D. VeteranGap', 'Dan Gap', 25, 'Italy', 'CB', 185.0, 81.0, 89, 76.0, 45.0, 72.0, 68.0, 90.0, 86.0),
                (9004, 2022, 'D. VeteranGap', 'Dan Gap', 28, 'Italy', 'CB', 185.0, 81.0, 89, 75.0, 45.0, 72.0, 68.0, 90.0, 86.0),
                -- 9005: Prime GK age 27-28 with NULL macro stats
                (9005, 2023, 'E. Keeper', 'Emil Keeper', 27, 'Germany', 'GK', 193.0, 86.0, 89, NULL, NULL, NULL, NULL, NULL, NULL),
                (9005, 2024, 'E. Keeper', 'Emil Keeper', 28, 'Germany', 'GK', 193.0, 86.0, 90, NULL, NULL, NULL, NULL, NULL, NULL);
            """
        )

        stg_cols = {
            "player_id": "INTEGER",
            "fifa_edition_year": "INTEGER",
            **{col: "DOUBLE" for col in ALL_SKILL_COLS},
        }
        ddl = ", ".join(f"{name} {dtype}" for name, dtype in stg_cols.items())
        con.execute(f"CREATE TABLE main.stg_fifa_granular ({ddl});")

        # Give 9001 distinct finishing/heading traits vs 9003 identical flat 75 vector
        s_9001 = [
            94.0
            if c in ("attacking_finishing", "power_shot_power", "movement_sprint_speed")
            else (10.0 if c.startswith("goalkeeping_") else 55.0)
            for c in ALL_SKILL_COLS
        ]
        s_9003 = [12.0 if c.startswith("goalkeeping_") else 75.0 for c in ALL_SKILL_COLS]
        placeholders = ", ".join(["?"] * (2 + len(ALL_SKILL_COLS)))
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});", [9001, 2022, *s_9001]
        )
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});", [9003, 2022, *s_9003]
        )
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});", [9003, 2023, *s_9003]
        )
    finally:
        con.close()


def test_discover_prime_candidates_filters_and_synthesizes_attributes(tmp_path: Path) -> None:
    """Qualified 2022-2024 consecutive prime stars are returned while gapped careers are skipped."""
    db_path = tmp_path / "promoter.duckdb"
    _create_synthetic_promoter_db(db_path)

    candidates = discover_prime_candidates(
        db_path=db_path,
        min_age=24,
        max_age=29,
        min_overall=88,
        min_consecutive_seasons=2,
    )
    pids = {c["player_id"] for c in candidates}
    assert pids == {9001, 9002, 9003, 9005}
    assert 9004 not in pids

    by_id = {c["player_id"]: c for c in candidates}
    assert by_id[9001]["peak_overall"] == 90
    assert by_id[9001]["attributes"]["attacking_finishing"] == 94.0

    # Modern 2024 CB without stg_fifa_granular row receives macro defending/pace attributes
    cb_attrs = by_id[9002]["attributes"]
    assert set(cb_attrs.keys()) == set(ALL_SKILL_COLS)
    assert cb_attrs["defending_standing_tackle"] == 91.0
    assert not any(np.isnan(v) for v in cb_attrs.values())

    # Prime GK receives >= 88 goalkeeping attributes despite NULL macro stats
    gk_attrs = by_id[9005]["attributes"]
    assert gk_attrs["goalkeeping_reflexes"] == 90.0


def test_novelty_gate_export_and_atomic_reload(tmp_path: Path) -> None:
    """Verify RBF novelty filtering, JSON/Parquet export, and zero-downtime registry reload."""
    db_path = tmp_path / "promoter.duckdb"
    _create_synthetic_promoter_db(db_path)

    # Seed base gallery containing flat-75 CM (matching 9003 clone) and 9005 already present
    base_gallery = [
        _sample_star(101, "Flat Midfielder", "CM", "Defensive & Central Midfielders", 90),
        _sample_star(9005, "E. Keeper", "GK", "Goalkeepers", 90),
    ]
    base_path = tmp_path / "superstar_gallery.json"
    base_path.write_text(json.dumps(base_gallery), encoding="utf-8")

    slot: AtomicArtifactSlot[ArchetypeVectorIndex] = AtomicArtifactSlot(
        ArchetypeVectorIndex(gallery_path=base_path)
    )
    registry = ModelRegistry(gallery_path=base_path)
    initial_idx = registry.get_archetype_indexer()
    assert len(initial_idx.gallery) == 2

    out_dir = tmp_path / "out"
    result = run_archetype_promotion_pipeline(
        db_path=db_path,
        base_gallery_path=base_path,
        output_dir=out_dir,
        version_tag="v2_test",
        min_rbf_distance=0.05,
        max_similarity_pct=98.0,
        write_parquet=True,
        slot=slot,
        registry=registry,
    )

    assert result.skipped_existing_count == 1  # 9005 already in gallery
    assert result.skipped_redundant_count == 1  # 9003 redundant vs player 101
    assert set(result.promoted_player_ids) == {9001, 9002}
    assert result.total_gallery_size == 4
    assert result.index_version_after_swap == 1

    json_file = Path(result.json_artifact_path)
    assert json_file.exists()
    assert len(json.loads(json_file.read_text(encoding="utf-8"))) == 4
    assert result.parquet_artifact_path is not None
    assert Path(result.parquet_artifact_path).exists()

    # Verify zero-downtime reload updated both slot and ModelRegistry cleanly
    reloaded = registry.get_archetype_indexer()
    assert reloaded is slot.get()
    assert len(reloaded.gallery) == 4
    assert reloaded is not initial_idx


def test_evaluate_novelty_and_hardware_guard(tmp_path: Path) -> None:
    """Verify evaluate_vector_novelty direct calls and hardware guard RSS check."""
    base_gallery = [
        _sample_star(101, "Baseline CM", "CM", "Defensive & Central Midfielders", 90),
    ]
    base_path = tmp_path / "gallery.json"
    base_path.write_text(json.dumps(base_gallery), encoding="utf-8")
    idx = ArchetypeVectorIndex(gallery_path=base_path)

    dup_eval = evaluate_vector_novelty(
        _sample_star(101, "Baseline CM", "CM", "Defensive & Central Midfielders", 90),
        idx,
    )
    assert dup_eval.is_novel is False
    assert dup_eval.reason == "already_in_gallery"

    novel_eval = evaluate_vector_novelty(
        _sample_star(
            202,
            "Distinct Striker",
            "ST",
            "Center Forwards & Strikers",
            91,
            attr_pattern={"attacking_finishing": 96.0, "defending_standing_tackle": 18.0},
        ),
        idx,
    )
    assert novel_eval.is_novel is True
    assert novel_eval.min_rbf_distance > 0.03

    # Atomic reload directly without slot/registry still loads fresh index
    fresh, swapped_ver = atomic_reload_archetype_index(base_path)
    assert fresh.is_loaded
    assert swapped_ver is None

    hw = apply_promoter_hardware_limits(max_memory_mb=1024.0)
    assert os.environ["DUCKDB_THREADS"] == "1"
    assert hw["rss_mb"] <= 1024.0
    with pytest.raises(MemoryError, match="exceeds promoter 1GB cap"):
        apply_promoter_hardware_limits(max_memory_mb=0.01)


def test_real_warehouse_promotion_sub_500ms_and_parquet(tmp_path: Path) -> None:
    """Execute end-to-end promotion against dbt_project/analytics_fc.duckdb in < 500ms."""
    real_db = Path("dbt_project/analytics_fc.duckdb")
    real_gallery = Path("data/processed/superstar_gallery.json")
    if not real_db.exists() or not real_gallery.exists():
        pytest.skip("Real DuckDB or superstar gallery unavailable.")

    result = run_archetype_promotion_pipeline(
        db_path=real_db,
        base_gallery_path=real_gallery,
        output_dir=tmp_path / "real_promo",
        version_tag="2022_2024",
        min_age=24,
        max_age=29,
        min_overall=88,
        min_consecutive_seasons=2,
        min_edition_year=2022,
        write_parquet=True,
    )

    assert result.execution_time_ms < 500.0
    assert result.candidates_evaluated > 0
    assert Path(result.json_artifact_path).exists()
    assert result.parquet_artifact_path is not None
    assert Path(result.parquet_artifact_path).exists()
    # Explicitly ensure empty-gallery fallback export does not fail
    empty_artifacts = export_versioned_gallery(
        [], output_dir=tmp_path / "empty_out", version_tag="v0"
    )
    assert empty_artifacts["json"].exists()
    assert empty_artifacts["parquet"].exists()


def test_empty_db_null_columns_empty_gallery_and_tmp_cleanup(tmp_path: Path) -> None:
    """Handle empty DBs, NULL/NaN strings or keys, empty [] galleries, and failed exports."""
    empty_db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(empty_db), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        con.execute(
            """
            CREATE TABLE main.fct_player_seasons (
                player_id INTEGER,
                fifa_edition_year INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                age INTEGER,
                nationality VARCHAR,
                positions VARCHAR,
                height_cm DOUBLE,
                weight_kg DOUBLE,
                overall INTEGER,
                pace DOUBLE,
                shooting DOUBLE,
                passing DOUBLE,
                dribbling DOUBLE,
                defending DOUBLE,
                physic DOUBLE
            );
            """
        )
    finally:
        con.close()

    assert discover_prime_candidates(db_path=empty_db) == []

    # Empty existing [] JSON file with slot & registry completes without ValueError
    empty_gallery_path = tmp_path / "empty_gallery.json"
    empty_gallery_path.write_text("[]", encoding="utf-8")
    slot: AtomicArtifactSlot[ArchetypeVectorIndex] = AtomicArtifactSlot()
    registry = ModelRegistry(gallery_path=empty_gallery_path)

    res_empty = run_archetype_promotion_pipeline(
        db_path=empty_db,
        base_gallery_path=empty_gallery_path,
        output_dir=tmp_path / "empty_run",
        version_tag="empty_test",
        slot=slot,
        registry=registry,
    )
    assert res_empty.total_gallery_size == 0
    assert res_empty.index_version_after_swap == 1
    assert slot.get() is not None
    assert registry.get_archetype_indexer().gallery == []

    # NULL key columns and NULL/mixed-NaN VARCHAR values must never crash or emit 'nan'
    null_db = tmp_path / "nulls.duckdb"
    con = duckdb.connect(str(null_db), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        con.execute(
            """
            CREATE TABLE main.fct_player_seasons (
                player_id INTEGER,
                fifa_edition_year INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                age INTEGER,
                nationality VARCHAR,
                positions VARCHAR,
                height_cm DOUBLE,
                weight_kg DOUBLE,
                overall INTEGER,
                pace DOUBLE,
                shooting DOUBLE,
                passing DOUBLE,
                dribbling DOUBLE,
                defending DOUBLE,
                physic DOUBLE
            );
            INSERT INTO main.fct_player_seasons VALUES
                (NULL, 2023, 'BadNullId', NULL, 25, NULL, 'ST', NULL, NULL, 90, 80, 80, 80, 80, 80, 80),
                (9101, NULL, 'BadNullYear', NULL, 25, NULL, 'ST', NULL, NULL, 90, 80, 80, 80, 80, 80, 80),
                (9102, 2022, 'MixedAnchor', 'Mixed Anchor', 25, 'Spain', 'ST', 180.0, 75.0, 89, 85, 85, 80, 82, 40, 80),
                (9103, 2022, NULL, NULL, 26, NULL, NULL, NULL, NULL, 89, 82, 82, 82, 82, 70, 80),
                (9103, 2023, NULL, NULL, 27, NULL, NULL, NULL, NULL, 90, 83, 83, 83, 83, 71, 81);
            """
        )
    finally:
        con.close()

    cands = discover_prime_candidates(
        db_path=null_db,
        min_age=24,
        max_age=29,
        min_overall=88,
        min_consecutive_seasons=2,
    )
    by_pid = {c["player_id"]: c for c in cands}
    assert set(by_pid.keys()) == {9103}
    c9103 = by_pid[9103]
    assert c9103["short_name"] == "9103"
    assert c9103["long_name"] == "9103"
    assert c9103["nationality"] == "Unknown"
    assert c9103["primary_position"] == "CM"
    assert c9103["positions"] == "CM"

    # Verify export_versioned_gallery cleans up all .tmp files and avoids split-brain JSON replace
    exp_dir = tmp_path / "fail_export"
    exp_dir.mkdir()
    preexisting_json = exp_dir / "superstar_gallery_vfail.json"
    preexisting_json.write_text('[{"keep": true}]', encoding="utf-8")

    with patch("src.models.archetype_promoter.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            export_versioned_gallery(
                [_sample_star(101, "Star", "CM", "Defensive & Central Midfielders", 90)],
                output_dir=exp_dir,
                version_tag="vfail",
                write_parquet=True,
            )

    assert list(exp_dir.glob("*.tmp")) == []
    assert json.loads(preexisting_json.read_text(encoding="utf-8")) == [{"keep": True}]
