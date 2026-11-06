WITH base_players AS (
    SELECT
        player_id,
        fifa_edition_year,
        fifa_version,
        short_name,
        long_name,
        age,
        date_of_birth,
        nationality,
        club,
        league_name,
        positions,
        preferred_foot,
        overall,
        potential,
        value_eur,
        wage_eur,
        pace,
        shooting,
        passing,
        dribbling,
        defending,
        physic
    FROM {{ ref('stg_modern_players') }}
),

windowed AS (
    SELECT
        -- Core Identity
        player_id,
        fifa_edition_year,
        fifa_version,
        short_name,
        long_name,
        age,
        date_of_birth,
        nationality,
        club,
        league_name,
        positions,
        preferred_foot,

        -- Primary Ratings & Potential Gap
        overall,
        potential,
        potential - overall AS potential_gap,
        value_eur,
        wage_eur,

        -- Career Indexing & Time-Series Anchors
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS career_season_index,

        COUNT(1) OVER (
            PARTITION BY player_id
        ) AS total_career_seasons,

        LAG(fifa_edition_year) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_fifa_edition_year,

        fifa_edition_year - LAG(fifa_edition_year) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS years_since_prev_season,

        -- Annual Rating Deltas
        LAG(overall) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_overall,

        overall - LAG(overall) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_overall,

        overall - FIRST_VALUE(overall) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS cumulative_overall_growth,

        LAG(potential) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_potential,

        potential - LAG(potential) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_potential,

        MAX(overall) OVER (
            PARTITION BY player_id
        ) AS career_peak_overall,

        -- Physical & Technical Attribute Deltas
        pace,
        pace - LAG(pace) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_pace,

        shooting,
        shooting - LAG(shooting) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_shooting,

        passing,
        passing - LAG(passing) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_passing,

        dribbling,
        dribbling - LAG(dribbling) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_dribbling,

        defending,
        defending - LAG(defending) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_defending,

        physic,
        physic - LAG(physic) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_physic,

        -- Financial Valuation Delta
        value_eur - LAG(value_eur) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS delta_value_eur
    FROM base_players
),

staged_metrics AS (
    SELECT
        *,
        -- Peak Performance Flag
        CASE
            WHEN overall = career_peak_overall THEN 1
            ELSE 0
        END AS is_peak_overall_season,

        -- Annual Developmental Velocity (Valid strictly for consecutive 1-year transitions)
        CASE
            WHEN years_since_prev_season = 1 THEN delta_overall
            ELSE NULL
        END AS annual_growth_velocity,

        -- Career Trajectory Classification
        CASE
            WHEN age <= 23 AND overall < potential THEN 'DEVELOPMENT'
            WHEN age BETWEEN 24 AND 29 THEN 'PRIME'
            WHEN age >= 30 THEN 'VETERAN'
            ELSE 'MATURE'
        END AS career_stage
    FROM windowed
)

SELECT * FROM staged_metrics
