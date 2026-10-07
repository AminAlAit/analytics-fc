WITH source AS (
    SELECT * FROM "analytics_fc"."raw"."tm_trophies"
),

renamed AS (
    SELECT
        short_name,
        club,
        team_trophy,
        individual_trophy,
        trophy,
        year AS trophy_year
    FROM source
)

SELECT * FROM renamed