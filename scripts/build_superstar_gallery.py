import os
import time
import json
import argparse
import subprocess
import duckdb
import pandas as pd
from pathlib import Path

# 34 granular skill attributes
SKILL_COLS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina",
    "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes"
]

SECTOR_MAP = {
    "GK": "Goalkeepers",
    "CB": "Defenders", "LB": "Defenders", "RB": "Defenders", 
    "LWB": "Defenders", "RWB": "Defenders",
    "CDM": "Defensive & Central Midfielders", "CM": "Defensive & Central Midfielders",
    "CAM": "Attacking Midfielders & Wingers", "LM": "Attacking Midfielders & Wingers", 
    "RM": "Attacking Midfielders & Wingers", "LW": "Attacking Midfielders & Wingers", 
    "RW": "Attacking Midfielders & Wingers",
    "CF": "Center Forwards & Strikers", "ST": "Center Forwards & Strikers"
}

def build_gallery(target_count=100, dvc_commit=False):
    print(f"[1/5] Initializing gentle environment for Top {target_count} personal peaks...")
    base_dir = Path(__file__).resolve().parent.parent
    db_path = str(base_dir / "dbt_project" / "analytics_fc.duckdb")
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time.sleep(1) # Gentle hardware pause

    print("[2/5] Connecting to DuckDB in single-threaded read-only mode...")
    con = duckdb.connect(db_path, read_only=True)
    con.execute("SET threads = 1;")
    con.execute("SET memory_limit = '1GB';")

    time.sleep(1)

    print(f"[3/5] Querying deterministic personal peak seasons for Top {target_count} players...")
    # Deterministic selection with tie-breakers: peak_overall DESC, seasons_at_peak DESC, max_rep DESC, player_id ASC
    query = f"""
    WITH player_career_summary AS (
        SELECT 
            player_id,
            MAX(overall) AS peak_overall,
            MAX(international_reputation) AS max_rep,
            COUNT(DISTINCT fifa_edition_year) AS total_career_seasons
        FROM stg_fifa_granular
        GROUP BY player_id
    ),
    
    peak_counts AS (
        SELECT 
            s.player_id,
            COUNT(*) AS seasons_at_peak
        FROM stg_fifa_granular s
        INNER JOIN player_career_summary p 
            ON s.player_id = p.player_id 
           AND s.overall = p.peak_overall
        GROUP BY s.player_id
    ),
    
    ranked_stars AS (
        SELECT 
            c.player_id, 
            c.peak_overall
        FROM player_career_summary c
        INNER JOIN peak_counts pc ON c.player_id = pc.player_id
        ORDER BY 
            c.peak_overall DESC, 
            pc.seasons_at_peak DESC, 
            c.max_rep DESC, 
            c.player_id ASC
        LIMIT {target_count}
    )
    
    SELECT 
        s.player_id,
        s.short_name,
        s.long_name,
        s.nationality,
        s.positions,
        s.age,
        s.fifa_edition_year,
        s.overall,
        s.height_cm,
        s.weight_kg,
        r.peak_overall,
        {', '.join(['s.' + col for col in SKILL_COLS])}
    FROM stg_fifa_granular s
    INNER JOIN ranked_stars r 
        ON s.player_id = r.player_id 
       AND s.overall = r.peak_overall
    ORDER BY r.peak_overall DESC, s.player_id, s.fifa_edition_year
    """
    
    df = con.execute(query).df()
    
    # Also fetch full career averages for composure backfilling where FIFA 15/16 lacked composure
    composure_fallback_query = """
    SELECT player_id, AVG(mentality_composure) as career_avg_composure
    FROM stg_fifa_granular
    WHERE mentality_composure IS NOT NULL
    GROUP BY player_id
    """
    composure_df = con.execute(composure_fallback_query).df().set_index("player_id")
    con.close()

    time.sleep(1) # Gentle hardware pause

    print(f"[4/5] Computing robust Personal Peak Centroids for {df['player_id'].nunique()} players...")
    
    gallery = []
    for player_id, group in df.groupby("player_id"):
        pos_list = [p.strip().upper() for p in str(group["positions"].iloc[0]).split(",")]
        primary_pos = pos_list[0]
        sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")
        
        # Build attribute dictionary with intelligent composure imputation for FIFA 15/16 veterans
        attributes = {}
        for col in SKILL_COLS:
            val_series = group[col].dropna()
            if not val_series.empty:
                attributes[col] = round(float(val_series.mean()), 2)
            elif col == "mentality_composure":
                # Backfill from other editions in their career, or derive from reactions/ball control
                if player_id in composure_df.index and pd.notna(composure_df.loc[player_id, "career_avg_composure"]):
                    attributes[col] = round(float(composure_df.loc[player_id, "career_avg_composure"]), 2)
                else:
                    # Conservative estimate based on technical reactions
                    reactions_val = group["movement_reactions"].dropna().mean()
                    attributes[col] = round(float(reactions_val if pd.notna(reactions_val) else 80.0), 2)
            else:
                attributes[col] = 50.0

        centroid = {
            "player_id": int(player_id),
            "short_name": group["short_name"].iloc[0],
            "long_name": group["long_name"].iloc[0],
            "nationality": group["nationality"].iloc[0],
            "primary_position": primary_pos,
            "positions": group["positions"].iloc[0],
            "tactical_sector": sector,
            "peak_overall": int(group["peak_overall"].iloc[0]),
            "peak_seasons_count": len(group),
            "peak_age_min": int(group["age"].min()),
            "peak_age_max": int(group["age"].max()),
            "peak_editions_observed": f"{group['fifa_edition_year'].min()}-{group['fifa_edition_year'].max()}",
            "height_cm": float(group["height_cm"].mean()),
            "weight_kg": float(group["weight_kg"].mean()),
            "attributes": attributes
        }
        gallery.append(centroid)

    # Sort descending by peak rating
    gallery.sort(key=lambda x: x["peak_overall"], reverse=True)

    output_file = output_dir / "superstar_gallery.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2)

    time.sleep(0.5)

    print(f"[5/5] Success! Generated {len(gallery)} fully-imputed Personal Peak Superstar Archetypes.")
    print(f"Saved gallery to: {output_file.name}")
    
    # Verify composure nulls
    null_composure_count = sum(1 for g in gallery if g["attributes"].get("mentality_composure") is None)
    print(f"Remaining null composure values: {null_composure_count}")

    if dvc_commit:
        print("[DVC] Automatically updating data/processed DVC tracking...")
        subprocess.run(["dvc", "add", "data/processed"], cwd=str(base_dir), check=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Top 100 Personal Peak Superstar Gallery")
    parser.add_argument("--count", type=int, default=100, help="Target superstar count (default: 100)")
    parser.add_argument("--dvc-commit", action="store_true", help="Automatically commit updated artifact to DVC")
    args = parser.parse_args()
    build_gallery(target_count=args.count, dvc_commit=args.dvc_commit)
