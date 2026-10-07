
  
  create view "analytics_fc"."main"."stg_tm_stats__dbt_tmp" as (
    ﻿WITH source AS (
    SELECT * FROM "analytics_fc"."raw"."tm_stats"
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
  );
