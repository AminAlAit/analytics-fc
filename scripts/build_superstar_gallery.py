import os
import time
import json
import duckdb
import pandas as pd
from pathlib import Path

def build_gallery():
    print("[1/5] Initializing gentle environment...")
    base_dir = Path(__file__).resolve().parent.parent
    db_path = str(base_dir / "dbt_project" / "analytics_fc.duckdb")
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pause to let CPU settle
    time.sleep(1)

    print("[2/5] Connecting to DuckDB in single-threaded read-only mode...")
    con = duckdb.connect(db_path, read_only=True)
    con.execute("SET threads = 1;")
    con.execute("SET memory_limit = '1GB';")

    time.sleep(1)

    # 34 granular skill attributes
    skill_cols = [
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

    print("[3/5] Querying canonical Superstar Gallery archetypes...")
    # SQL query enforcing our 4 objective specification filters
    query = f"""
    WITH prime_seasons AS (
        SELECT 
            player_id,
            short_name,
            long_name,
            nationality,
            positions,
            age,
            fifa_edition_year,
            overall,
            potential,
            international_reputation,
            height_cm,
            weight_kg,
            {', '.join(skill_cols)}
        FROM stg_fifa_granular
        WHERE age BETWEEN 26 AND 29
    ),
    
    qualifying_players AS (
        SELECT 
            player_id,
            short_name,
            MAX(overall) AS peak_overall,
            COUNT(DISTINCT fifa_edition_year) AS prime_seasons_observed,
            SUM(CASE WHEN overall >= 85 THEN 1 ELSE 0 END) AS seasons_above_85,
            SUM(CASE WHEN overall >= 88 THEN 1 ELSE 0 END) AS seasons_above_88,
            MAX(international_reputation) AS max_rep
        FROM prime_seasons
        GROUP BY player_id, short_name
        HAVING 
            MAX(overall) >= 88
            AND MAX(international_reputation) >= 3
            AND (
                SUM(CASE WHEN overall >= 85 THEN 1 ELSE 0 END) >= 3
                OR SUM(CASE WHEN overall >= 88 THEN 1 ELSE 0 END) >= 2
            )
    )
    
    SELECT 
        p.*,
        q.peak_overall
    FROM prime_seasons p
    INNER JOIN qualifying_players q ON p.player_id = q.player_id
    ORDER BY q.peak_overall DESC, p.player_id, p.fifa_edition_year
    """
    
    df = con.execute(query).df()
    con.close()

    time.sleep(1)

    print(f"[4/5] Computing Prime Centroid vectors across {df['player_id'].nunique()} qualifying superstars...")
    
    gallery = []
    for player_id, group in df.groupby("player_id"):
        # Primary tactical sector
        pos_list = str(group["positions"].iloc[0]).split(",")
        primary_pos = pos_list[0].strip().upper()
        
        # Categorize tactical sector
        if primary_pos in ["GK"]:
            sector = "Goalkeepers"
        elif primary_pos in ["CB", "LB", "RB", "LWB", "RWB"]:
            sector = "Defenders"
        elif primary_pos in ["CDM", "CM"]:
            sector = "Defensive & Central Midfielders"
        elif primary_pos in ["CAM", "LM", "RM", "LW", "RW"]:
            sector = "Attacking Midfielders & Wingers"
        else:
            sector = "Center Forwards & Strikers"

        # Compute centroid (mean across prime seasons)
        centroid = {
            "player_id": int(player_id),
            "short_name": group["short_name"].iloc[0],
            "long_name": group["long_name"].iloc[0],
            "nationality": group["nationality"].iloc[0],
            "primary_position": primary_pos,
            "positions": group["positions"].iloc[0],
            "tactical_sector": sector,
            "peak_overall": int(group["peak_overall"].iloc[0]),
            "prime_seasons_count": len(group),
            "prime_age_window": f"{group['age'].min()}-{group['age'].max()}",
            "height_cm": float(group["height_cm"].mean()),
            "weight_kg": float(group["weight_kg"].mean()),
            "attributes": {
                col: round(float(group[col].dropna().mean()), 2) if not group[col].dropna().empty else None 
                for col in skill_cols
            }
        }
        gallery.append(centroid)

    # Sort by peak overall descending
    gallery.sort(key=lambda x: x["peak_overall"], reverse=True)

    output_file = output_dir / "superstar_gallery.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2)

    print(f"[5/5] Successfully generated Superstar Gallery!")
    print(f"Saved {len(gallery)} canonical superstars to: {output_file.name}")
    print("\nTop 10 Benchmark Superstars:")
    for i, star in enumerate(gallery[:10], 1):
        print(f"  {i}. {star['short_name']} ({star['tactical_sector']}) - Peak: {star['peak_overall']} OVR [{star['prime_age_window']}]")

if __name__ == "__main__":
    build_gallery()
