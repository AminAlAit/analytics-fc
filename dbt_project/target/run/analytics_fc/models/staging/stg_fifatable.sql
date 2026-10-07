
  
  create view "analytics_fc"."main"."stg_fifatable__dbt_tmp" as (
    ﻿WITH source AS (
    SELECT * FROM "analytics_fc"."raw"."fifatable"
),

renamed AS (
    SELECT
        player_id,
        Fifa_year AS fifa_edition_year,
        age,
        short_name,
        long_name,
        nationality,
        club,
        position,
        overall,
        max_potential
    FROM source
)

SELECT * FROM renamed
  );
