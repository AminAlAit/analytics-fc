WITH source AS (
    SELECT * FROM {{ source('raw', 'granular_15_24') }}
),

renamed AS (
    SELECT
        -- Identity
        TRY_CAST(player_id AS INTEGER) AS player_id,
        2000 + TRY_CAST(fifa_version AS INTEGER) AS fifa_edition_year,
        TRY_CAST(fifa_version AS INTEGER) AS fifa_version,
        TRY_CAST(fifa_update AS INTEGER) AS fifa_update,
        short_name,
        long_name,
        TRY_CAST(age AS INTEGER) AS age,
        TRY_CAST(dob AS DATE) AS date_of_birth,
        nationality_name AS nationality,
        TRY_CAST(nationality_id AS INTEGER) AS nationality_id,
        club_name AS club,
        TRY_CAST(club_team_id AS INTEGER) AS club_team_id,
        league_name,
        TRY_CAST(league_id AS INTEGER) AS league_id,
        TRY_CAST(league_level AS INTEGER) AS league_level,
        player_positions AS positions,
        preferred_foot,
        TRY_CAST(height_cm AS INTEGER) AS height_cm,
        TRY_CAST(weight_kg AS INTEGER) AS weight_kg,

        -- Headline ratings
        TRY_CAST(overall AS INTEGER) AS overall,
        TRY_CAST(potential AS INTEGER) AS potential,
        TRY_CAST(value_eur AS DOUBLE) AS value_eur,
        TRY_CAST(wage_eur AS DOUBLE) AS wage_eur,
        TRY_CAST(release_clause_eur AS DOUBLE) AS release_clause_eur,
        TRY_CAST(international_reputation AS INTEGER) AS international_reputation,
        TRY_CAST(weak_foot AS INTEGER) AS weak_foot,
        TRY_CAST(skill_moves AS INTEGER) AS skill_moves,

        -- Face attributes (null for goalkeepers in modern editions)
        TRY_CAST(pace AS INTEGER) AS pace,
        TRY_CAST(shooting AS INTEGER) AS shooting,
        TRY_CAST(passing AS INTEGER) AS passing,
        TRY_CAST(dribbling AS INTEGER) AS dribbling,
        TRY_CAST(defending AS INTEGER) AS defending,
        TRY_CAST(physic AS INTEGER) AS physic,

        -- Detailed skill attributes
        TRY_CAST(attacking_crossing AS INTEGER) AS attacking_crossing,
        TRY_CAST(attacking_finishing AS INTEGER) AS attacking_finishing,
        TRY_CAST(attacking_heading_accuracy AS INTEGER) AS attacking_heading_accuracy,
        TRY_CAST(attacking_short_passing AS INTEGER) AS attacking_short_passing,
        TRY_CAST(attacking_volleys AS INTEGER) AS attacking_volleys,
        TRY_CAST(skill_dribbling AS INTEGER) AS skill_dribbling,
        TRY_CAST(skill_curve AS INTEGER) AS skill_curve,
        TRY_CAST(skill_fk_accuracy AS INTEGER) AS skill_fk_accuracy,
        TRY_CAST(skill_long_passing AS INTEGER) AS skill_long_passing,
        TRY_CAST(skill_ball_control AS INTEGER) AS skill_ball_control,
        TRY_CAST(movement_acceleration AS INTEGER) AS movement_acceleration,
        TRY_CAST(movement_sprint_speed AS INTEGER) AS movement_sprint_speed,
        TRY_CAST(movement_agility AS INTEGER) AS movement_agility,
        TRY_CAST(movement_reactions AS INTEGER) AS movement_reactions,
        TRY_CAST(movement_balance AS INTEGER) AS movement_balance,
        TRY_CAST(power_shot_power AS INTEGER) AS power_shot_power,
        TRY_CAST(power_jumping AS INTEGER) AS power_jumping,
        TRY_CAST(power_stamina AS INTEGER) AS power_stamina,
        TRY_CAST(power_strength AS INTEGER) AS power_strength,
        TRY_CAST(power_long_shots AS INTEGER) AS power_long_shots,
        TRY_CAST(mentality_aggression AS INTEGER) AS mentality_aggression,
        TRY_CAST(mentality_interceptions AS INTEGER) AS mentality_interceptions,
        TRY_CAST(mentality_positioning AS INTEGER) AS mentality_positioning,
        TRY_CAST(mentality_vision AS INTEGER) AS mentality_vision,
        TRY_CAST(mentality_penalties AS INTEGER) AS mentality_penalties,
        TRY_CAST(mentality_composure AS INTEGER) AS mentality_composure,
        TRY_CAST(defending_marking_awareness AS INTEGER) AS defending_marking_awareness,
        TRY_CAST(defending_standing_tackle AS INTEGER) AS defending_standing_tackle,
        TRY_CAST(defending_sliding_tackle AS INTEGER) AS defending_sliding_tackle,
        TRY_CAST(goalkeeping_diving AS INTEGER) AS goalkeeping_diving,
        TRY_CAST(goalkeeping_handling AS INTEGER) AS goalkeeping_handling,
        TRY_CAST(goalkeeping_kicking AS INTEGER) AS goalkeeping_kicking,
        TRY_CAST(goalkeeping_positioning AS INTEGER) AS goalkeeping_positioning,
        TRY_CAST(goalkeeping_reflexes AS INTEGER) AS goalkeeping_reflexes,
        TRY_CAST(goalkeeping_speed AS INTEGER) AS goalkeeping_speed
    FROM source
)

SELECT * FROM renamed
