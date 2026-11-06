-- Asserts that there are zero orphaned seasonal records in fct_player_seasons
-- that do not have a matching parent record in dim_players.
-- Any returned rows indicate a referential integrity violation.

SELECT
    f.player_id,
    f.fifa_edition_year,
    f.short_name
FROM {{ ref('fct_player_seasons') }} f
LEFT JOIN {{ ref('dim_players') }} d
    ON f.player_id = d.player_id
WHERE d.player_id IS NULL
