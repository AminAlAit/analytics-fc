
  
  create view "analytics_fc"."main"."stg_fifatable__dbt_tmp" as (
    ﻿WITH source AS (
    SELECT * FROM "analytics_fc"."raw"."fifatable"
),

renamed AS (
    SELECT
        player_id,
        Fifa_year AS fifa_edition_year,
        TRY_CAST(age AS INTEGER) AS age,
        short_name,
        long_name,
        nationality,
        club,
        position,
        TRY_CAST(overall AS INTEGER) AS overall,
        TRY_CAST(max_potential AS INTEGER) AS max_potential,
        TRY_CAST(minutes AS INTEGER) AS minutes,
        TRY_CAST(goals AS INTEGER) AS goals
    FROM source
)

SELECT * FROM renamed
  );
