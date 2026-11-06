WITH player_seasons AS (
    SELECT
        player_id,
        fifa_edition_year,
        short_name,
        long_name,
        date_of_birth,
        nationality,
        preferred_foot,
        height_cm,
        weight_kg,
        club,
        league_name,
        positions,
        overall,
        potential,
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year DESC
        ) AS recency_rank,
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS rookie_rank
    FROM {{ ref('stg_modern_players') }}
),

career_aggregates AS (
    SELECT
        player_id,
        MIN(fifa_edition_year) AS rookie_edition_year,
        MAX(fifa_edition_year) AS latest_edition_year,
        COUNT(DISTINCT fifa_edition_year) AS total_seasons_recorded,
        MAX(overall) AS career_peak_overall
    FROM {{ ref('stg_modern_players') }}
    GROUP BY player_id
),

rookie_info AS (
    SELECT
        player_id,
        overall AS debut_overall,
        potential AS debut_potential
    FROM player_seasons
    WHERE rookie_rank = 1
),

latest_info AS (
    SELECT
        player_id,
        short_name,
        long_name,
        date_of_birth,
        nationality,
        preferred_foot,
        height_cm,
        weight_kg,
        club AS latest_club,
        league_name AS latest_league,
        positions AS latest_positions
    FROM player_seasons
    WHERE recency_rank = 1
),

final_dim AS (
    SELECT
        l.player_id,
        l.short_name,
        l.long_name,
        l.date_of_birth,
        l.nationality,
        l.preferred_foot,
        l.height_cm,
        l.weight_kg,
        l.latest_club,
        l.latest_league,
        l.latest_positions,
        c.rookie_edition_year,
        c.latest_edition_year,
        c.total_seasons_recorded,
        r.debut_overall,
        c.career_peak_overall,
        r.debut_potential,
        c.career_peak_overall - r.debut_overall AS lifetime_growth
    FROM latest_info l
    INNER JOIN career_aggregates c ON l.player_id = c.player_id
    INNER JOIN rookie_info r ON l.player_id = r.player_id
)

SELECT * FROM final_dim
