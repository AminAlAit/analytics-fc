WITH source AS (
    SELECT * FROM {{ source('raw', 'fifatable') }}
),

renamed AS (
    SELECT
        player_id,
        Fifa_year AS fifa_edition_year,
        CAST(age AS INTEGER) AS age,
        short_name,
        long_name,
        nationality,
        club,
        position,
        CAST(overall AS INTEGER) AS overall,
        CAST(max_potential AS INTEGER) AS max_potential
    FROM source
)

SELECT * FROM renamed
