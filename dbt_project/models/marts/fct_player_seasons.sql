WITH growth_curves AS (
    SELECT
        player_id,
        fifa_edition_year,
        career_season_index,
        total_career_seasons,
        prev_fifa_edition_year,
        years_since_prev_season,
        prev_overall,
        delta_overall,
        cumulative_overall_growth,
        prev_potential,
        delta_potential,
        career_peak_overall,
        is_peak_overall_season,
        annual_growth_velocity,
        career_stage,
        delta_pace,
        delta_shooting,
        delta_passing,
        delta_dribbling,
        delta_defending,
        delta_physic,
        delta_value_eur
    FROM {{ ref('int_player_growth_curves') }}
),

staged_seasons AS (
    SELECT
        player_id,
        fifa_edition_year,
        fifa_version,
        short_name,
        long_name,
        age,
        date_of_birth,
        nationality,
        nationality_id,
        club,
        club_team_id,
        league_name,
        league_id,
        league_level,
        positions,
        preferred_foot,
        height_cm,
        weight_kg,
        overall,
        potential,
        value_eur,
        wage_eur,
        release_clause_eur,
        international_reputation,
        weak_foot,
        skill_moves,
        pace,
        shooting,
        passing,
        dribbling,
        defending,
        physic
    FROM {{ ref('stg_modern_players') }}
),

final_fact AS (
    SELECT
        -- Primary Composite Grain & Foreign Keys
        s.player_id,
        s.fifa_edition_year,
        s.fifa_version,

        -- Player Demographics & Club Context
        s.short_name,
        s.long_name,
        s.age,
        s.date_of_birth,
        s.nationality,
        s.nationality_id,
        s.club,
        s.club_team_id,
        s.league_name,
        s.league_id,
        s.league_level,
        s.positions,
        s.preferred_foot,
        s.height_cm,
        s.weight_kg,

        -- Headline Performance & Market Valuation
        s.overall,
        s.potential,
        s.potential - s.overall AS potential_gap,
        s.value_eur,
        s.wage_eur,
        s.release_clause_eur,
        s.international_reputation,
        s.weak_foot,
        s.skill_moves,

        -- Face Attributes
        s.pace,
        s.shooting,
        s.passing,
        s.dribbling,
        s.defending,
        s.physic,

        -- Longitudinal Career Window Metrics
        g.career_season_index,
        g.total_career_seasons,
        g.prev_fifa_edition_year,
        g.years_since_prev_season,
        g.prev_overall,
        g.delta_overall,
        g.cumulative_overall_growth,
        g.prev_potential,
        g.delta_potential,
        g.career_peak_overall,
        g.is_peak_overall_season,
        g.annual_growth_velocity,
        g.career_stage,

        -- Attribute Growth Deltas
        g.delta_pace,
        g.delta_shooting,
        g.delta_passing,
        g.delta_dribbling,
        g.delta_defending,
        g.delta_physic,
        g.delta_value_eur
    FROM staged_seasons s
    INNER JOIN growth_curves g
        ON s.player_id = g.player_id
        AND s.fifa_edition_year = g.fifa_edition_year
)

SELECT * FROM final_fact
