WITH source AS (
    SELECT * FROM {{ source('raw', 'tm_stats') }}
),

renamed AS (
    SELECT
        FifaIndex AS fifa_index,
        short_name,
        competition,
        club,
        games_played,
        goals,
        assists,
        minutes
    FROM source
)

SELECT * FROM renamed
