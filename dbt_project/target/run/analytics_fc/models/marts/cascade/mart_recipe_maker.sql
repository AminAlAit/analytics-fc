
    

    create  table
      "analytics_fc"."main"."mart_recipe_maker__dbt_tmp"
  
    
    as (
      ﻿-- Level 3: Recipe Maker
-- This model takes the validated Patterns from Level 2 and anchors them to the 
-- "Silverware Event" (winning a trophy). It calculates the empirical probability 
-- that a prospect will trigger the event based on the success rate of their historical matches.

WITH patterns AS (
    SELECT 
        player_a_id AS prospect_id,
        player_a_name AS prospect_name,
        player_b_id AS historical_match_id,
        player_b_name AS historical_match_name,
        scdi_overall,
        scdi_potential,
        scdi_minutes
    FROM "analytics_fc"."main"."mart_pattern_maker"
),

-- The Terminating Event: Winning a trophy
trophy_events AS (
    SELECT DISTINCT short_name
    FROM "analytics_fc"."main"."stg_tm_trophies"
),

-- Check if the historical match (Ingredient) successfully triggered the Event
historical_outcomes AS (
    SELECT 
        p.*,
        CASE WHEN t.short_name IS NOT NULL THEN 1 ELSE 0 END AS achieved_silverware_event
    FROM patterns p
    LEFT JOIN trophy_events t ON p.historical_match_name = t.short_name
)

-- Recipe Aggregation: Calculate empirical probability
SELECT 
    prospect_id,
    prospect_name,
    COUNT(historical_match_id) AS total_historical_patterns,
    SUM(achieved_silverware_event) AS patterns_leading_to_event,
    ROUND(SUM(achieved_silverware_event) * 100.0 / COUNT(historical_match_id), 2) AS empirical_probability_percentage
FROM historical_outcomes
GROUP BY prospect_id, prospect_name
-- Require at least 3 historical matches to form a valid Recipe
HAVING COUNT(historical_match_id) >= 3
ORDER BY empirical_probability_percentage DESC, total_historical_patterns DESC
    );
    
  