
    

    create  table
      "analytics_fc"."main"."mart_scdi_overall__dbt_tmp"
  
    
    as (
      ﻿-- Level 1: Single Correlating Dyadic Index (SCDI)
-- This model finds pairs of players whose 'overall' rating trajectories 
-- highly correlate during their developmental years (ages 18 to 21).

WITH base AS (
    SELECT 
        player_id, 
        short_name,
        age, 
        overall
    FROM "analytics_fc"."main"."stg_fifatable"
    WHERE age BETWEEN 18 AND 21
),

-- Ensure we only compare players who actually have data for all 4 years in this window
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

-- Join every player with every other player based on the same age
trajectory_pairs AS (
    SELECT 
        a.player_id AS player_a_id,
        a.short_name AS player_a_name,
        b.player_id AS player_b_id,
        b.short_name AS player_b_name,
        a.age,
        a.overall AS overall_a,
        b.overall AS overall_b
    FROM filtered_base a
    INNER JOIN filtered_base b ON a.age = b.age
    WHERE a.player_id < b.player_id -- ensures we don't compare A to A, or do A->B and B->A
)

-- Calculate the Pearson Correlation (SCDI) for the trajectory
SELECT 
    player_a_id,
    player_a_name,
    player_b_id,
    player_b_name,
    CORR(overall_a, overall_b) AS scdi_score
FROM trajectory_pairs
GROUP BY 
    player_a_id, player_a_name, player_b_id, player_b_name
-- Filter for only highly correlated pairs (the "gems")
HAVING CORR(overall_a, overall_b) >= 0.85
    );
    
  