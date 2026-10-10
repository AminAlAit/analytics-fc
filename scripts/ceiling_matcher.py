import json
import time
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# Tactical sector mapping for positional partitioning
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

# Outfield-relevant features (29 skill traits + physical mechanics)
OUTFIELD_COLS = [col for col in SKILL_COLS if not col.startswith("goalkeeping_")]

class SuperstarCeilingMatcher:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.db_path = str(base_dir / "dbt_project" / "analytics_fc.duckdb")
        gallery_path = base_dir / "data" / "processed" / "superstar_gallery.json"
        
        with open(gallery_path, "r", encoding="utf-8") as f:
            self.gallery = json.load(f)
            
    def _get_con(self):
        con = duckdb.connect(self.db_path, read_only=True)
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        return con

    def match_prospect(self, player_name, edition_year=None, age_max=21, top_k=3):
        con = self._get_con()
        time.sleep(0.5)  # Gentle hardware pause
        
        # Query prospect record at young age
        year_filter = f"AND fifa_edition_year = {edition_year}" if edition_year else ""
        query = f"""
        SELECT 
            player_id, short_name, long_name, age, fifa_edition_year,
            overall, potential, positions, height_cm, weight_kg,
            {', '.join(SKILL_COLS)}
        FROM stg_fifa_granular
        WHERE lower(short_name) LIKE lower('%{player_name}%')
          AND age <= {age_max}
          {year_filter}
        ORDER BY overall DESC
        LIMIT 1
        """
        prospect_df = con.execute(query).df()
        con.close()
        
        if prospect_df.empty:
            return {"error": f"No youth profile (age <= {age_max}) found for '{player_name}'"}
            
        prospect = prospect_df.iloc[0]
        pos_list = [p.strip().upper() for p in str(prospect["positions"]).split(",")]
        primary_pos = pos_list[0]
        sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")
        is_gk = (primary_pos == "GK")
        
        # Feature columns to evaluate
        active_cols = [c for c in SKILL_COLS if c.startswith("goalkeeping_")] if is_gk else OUTFIELD_COLS
        
        # Construct prospect vector, handling any missing values gracefully
        p_vec = np.array([
            float(prospect[c]) if pd.notna(prospect[c]) else 50.0 
            for c in active_cols
        ])
        
        # L2 Directional Normalization (pure playstyle geometry)
        p_norm = np.linalg.norm(p_vec)
        p_unit = p_vec / p_norm if p_norm > 0 else p_vec
        
        # Filter candidates in the same sector (or adjacent offensive sectors if striker/winger)
        candidates = []
        for star in self.gallery:
            # Exclude self-matching to ensure genuine external archetype comparisons
            if star["player_id"] == int(prospect["player_id"]):
                continue

            star_sector = star["tactical_sector"]
            # Allow positional fluidity between wingers and strikers
            sector_match = (star_sector == sector) or (
                sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"] and
                star_sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"]
            )
            
            if is_gk and star_sector != "Goalkeepers":
                continue
            if not is_gk and star_sector == "Goalkeepers":
                continue
            if not sector_match:
                continue
                
            s_vec = np.array([
                float(star["attributes"].get(c, 50.0)) if star["attributes"].get(c) is not None else 50.0 
                for c in active_cols
            ])
            s_norm = np.linalg.norm(s_vec)
            s_unit = s_vec / s_norm if s_norm > 0 else s_vec
            
            # Cosine similarity on playstyle shape
            cos_sim = float(np.dot(p_unit, s_unit))
            
            # Normalized Euclidean distance
            norm_dist = float(np.linalg.norm(p_unit - s_unit))
            
            # Calibrated Similarity Percentage (Gaussian RBF Kernel)
            calibrated_pct = round(float(np.exp(- (norm_dist ** 2) / (2 * (0.28 ** 2))) * 100), 1)
            
            # Trait gap analysis
            trait_deltas = {}
            for col in active_cols:
                s_val = star["attributes"].get(col)
                p_val = prospect[col]
                if s_val is not None and pd.notna(p_val):
                    trait_deltas[col] = round(float(s_val) - float(p_val), 1)
                    
            # Sort deltas: biggest development needed vs already matched
            top_growth_needed = sorted(
                [{"attribute": k, "gap": v} for k, v in trait_deltas.items() if v > 0],
                key=lambda x: x["gap"], reverse=True
            )[:3]
            
            strengths_aligned = sorted(
                [{"attribute": k, "difference": abs(v)} for k, v in trait_deltas.items()],
                key=lambda x: x["difference"]
            )[:3]

            peak_window = f"{star.get('peak_age_min', 26)}-{star.get('peak_age_max', 29)}"
            candidates.append({
                "superstar_name": star["short_name"],
                "tactical_sector": star["tactical_sector"],
                "peak_overall": star["peak_overall"],
                "prime_age_window": peak_window,
                "similarity_score_pct": calibrated_pct,
                "top_growth_priorities": top_growth_needed,
                "closest_matching_traits": strengths_aligned
            })
            
        candidates.sort(key=lambda x: x["similarity_score_pct"], reverse=True)
        top_matches = candidates[:top_k]
        
        return {
            "prospect": {
                "player_id": int(prospect["player_id"]),
                "name": prospect["short_name"],
                "age_at_evaluation": int(prospect["age"]),
                "fifa_edition": int(prospect["fifa_edition_year"]),
                "overall": int(prospect["overall"]),
                "potential": int(prospect["potential"]),
                "positions": prospect["positions"],
                "tactical_sector": sector
            },
            "matches": top_matches
        }

if __name__ == "__main__":
    print("Testing Superstar Ceiling Matcher on historical wonderkids...")
    matcher = SuperstarCeilingMatcher()
    
    test_prospects = ["K. Mbappé", "E. Haaland", "Pedri", "T. Alexander-Arnold"]
    for name in test_prospects:
        time.sleep(1) # Gentle pause between tests
        res = matcher.match_prospect(name)
        if "error" in res:
            print(f"\n[!] {res['error']}")
            continue
            
        p = res["prospect"]
        safe_p_name = p['name'].encode('ascii', 'replace').decode()
        print(f"\n==================================================")
        print(f"PROSPECT: {safe_p_name} (Age {p['age_at_evaluation']} in FIFA {p['fifa_edition']})")
        print(f"Position: {p['positions']} | Rating: {p['overall']} OVR (Potential: {p['potential']})")
        print(f"Tactical Sector: {p['tactical_sector']}")
        print("--------------------------------------------------")
        print("PROJECTED SUPERSTAR CEILINGS:")
        for i, m in enumerate(res["matches"], 1):
            safe_s_name = m['superstar_name'].encode('ascii', 'replace').decode()
            print(f"  #{i} {safe_s_name} ({m['peak_overall']} Peak OVR [{m['prime_age_window']}])")
            print(f"     Playstyle Match: {m['similarity_score_pct']}%")
            traits = ", ".join([t['attribute'].replace('skill_', '').replace('movement_', '').replace('mentality_', '') for t in m['closest_matching_traits']])
            print(f"     Closest Traits: {traits}")
            gaps = ", ".join([f"{g['attribute'].replace('attacking_', '').replace('mentality_', '')} (+{g['gap']})" for g in m['top_growth_priorities'][:2]])
            print(f"     Growth Needed:  {gaps}")
