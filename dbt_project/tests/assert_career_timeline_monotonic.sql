-- Asserts that career timeline progression is strictly monotonic across seasons:
-- 1. fifa_edition_year must be strictly increasing per player.
-- 2. career_season_index must be strictly increasing per player.
-- 3. age must be non-decreasing across consecutive seasons for a consistent date_of_birth.
-- Any returned rows indicate timeline corruption or inverted chronological order.

WITH season_lagged AS (
    SELECT
        player_id,
        fifa_edition_year,
        prev_fifa_edition_year,
        career_season_index,
        LAG(career_season_index) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_career_season_index,
        age,
        LAG(age) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_age,
        date_of_birth,
        LAG(date_of_birth) OVER (
            PARTITION BY player_id
            ORDER BY fifa_edition_year ASC
        ) AS prev_date_of_birth
    FROM {{ ref('fct_player_seasons') }}
)

SELECT
    player_id,
    fifa_edition_year,
    prev_fifa_edition_year,
    career_season_index,
    prev_career_season_index,
    age,
    prev_age
FROM season_lagged
WHERE prev_fifa_edition_year IS NOT NULL
  AND (
      fifa_edition_year <= prev_fifa_edition_year
      OR career_season_index <= prev_career_season_index
      OR (date_of_birth = prev_date_of_birth AND age < prev_age)
  )
