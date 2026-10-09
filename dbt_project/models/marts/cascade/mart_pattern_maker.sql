WITH base AS (
    SELECT 
        player_id, 
        short_name,
        age, 
        overall,
        max_potential,
        minutes
    FROM {{ ref('stg_fifatable') }}
    WHERE age BETWEEN 19 AND 22
),

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
    CASE 
        WHEN stddev_pop(overall_a) = 0 AND stddev_pop(overall_b) = 0 THEN 1.0
        WHEN stddev_pop(overall_a) = 0 OR stddev_pop(overall_b) = 0 THEN 0.0
        ELSE CORR(overall_a, overall_b)
    END AS scdi_overall,
    CASE 
        WHEN stddev_pop(pot_a) = 0 AND stddev_pop(pot_b) = 0 THEN 1.0
        WHEN stddev_pop(pot_a) = 0 OR stddev_pop(pot_b) = 0 THEN 0.0
        ELSE CORR(pot_a, pot_b)
    END AS scdi_potential,
    NULL AS scdi_minutes
FROM trajectory_pairs
GROUP BY 
    player_a_id, player_a_name, player_b_id, player_b_name
HAVING 
    (
        CASE 
            WHEN stddev_pop(overall_a) = 0 AND stddev_pop(overall_b) = 0 THEN 1.0
            WHEN stddev_pop(overall_a) = 0 OR stddev_pop(overall_b) = 0 THEN 0.0
            ELSE CORR(overall_a, overall_b)
        END
    ) >= 0.85
    AND 
    (
        CASE 
            WHEN stddev_pop(pot_a) = 0 AND stddev_pop(pot_b) = 0 THEN 1.0
            WHEN stddev_pop(pot_a) = 0 OR stddev_pop(pot_b) = 0 THEN 0.0
            ELSE CORR(pot_a, pot_b)
        END
    ) >= 0.80
