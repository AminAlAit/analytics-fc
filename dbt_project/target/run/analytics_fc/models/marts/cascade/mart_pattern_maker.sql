
    

    create  table
      "analytics_fc"."main"."mart_pattern_maker__dbt_tmp"
  
    
    as (
      ﻿-- Level 1 & 2: Pattern Maker (Multi-Indicator SCDIs)
-- Calculates the Pearson Correlation across multiple traits (Overall, Potential, Minutes)
-- and filters for pairs that strongly correlate on ALL indicators.

WITH base AS (
    SELECT 
        player_id, 
        short_name,
        age, 
        overall,
        max_potential,
        minutes
    FROM "analytics_fc"."main"."stg_fifatable"
    WHERE age BETWEEN 18 AND 21
),

-- Ensure 4 years of data
complete_trajectories AS (
    SELECT player_id
    FROM base
    GROUP BY player_id
    HAVING COUNT(*) = 4
),

filtered_base AS (
    SELECT b.*
    FROM base b
    INNER JOIN complete_trajectories ct ON b.player_id = ct.player_id
),

trajectory_pairs AS (
    SELECT 
        a.player_id AS player_a_id,
        a.short_name AS player_a_name,
        b.player_id AS player_b_id,
        b.short_name AS player_b_name,
        a.age,
        a.overall AS overall_a, b.overall AS overall_b,
        a.max_potential AS pot_a, b.max_potential AS pot_b,
        a.minutes AS min_a, b.minutes AS min_b
    FROM filtered_base a
    INNER JOIN filtered_base b ON a.age = b.age
    WHERE a.player_id < b.player_id
)

SELECT 
    player_a_id,
    player_a_name,
    player_b_id,
    player_b_name,
    CORR(overall_a, overall_b) AS scdi_overall,
    CORR(pot_a, pot_b) AS scdi_potential,
    CORR(min_a, min_b) AS scdi_minutes
FROM trajectory_pairs
GROUP BY 
    player_a_id, player_a_name, player_b_id, player_b_name
-- Level 2 Pattern Filter: Must strongly correlate across multiple traits
HAVING CORR(overall_a, overall_b) >= 0.85
   AND CORR(pot_a, pot_b) >= 0.80
    );
    
  