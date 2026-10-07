# Analytics FC Methodology Charter

## The Essence
Analytics FC builds **recipes** that describe how a football player's developmental indicator movements have historically led to specific career events. It then scores every young prospect by how closely their recent indicator movements match each recipe's canonical shape; the closer the match, the higher the empirical probability of that event occurring.

## The Methodological Hierarchy
This project strictly adheres to the Correlation Cascade methodology, built upon the following hierarchy:

1. **Indicators:** The raw, quantitative metrics (e.g., overall rating, minutes played, max_potential).
2. **SCDIs (Single Correlating Dyadic Indexes):** The mathematical Pearson correlation score between two players on a *single* indicator over a fixed canonical length (e.g., Player A and Player B have an SCDI of 0.90 on their overall trajectory from age 18 to 21).
3. **The Pattern:** The synthesis of multiple high-scoring SCDIs across different indicators simultaneously. A Pattern exists only when two players strongly correlate across multiple independent indicators (e.g., overall AND minutes AND potential) during the same developmental window.
4. **The Ingredients:** Historical Patterns that precede and terminate in a verified target Event.
5. **The Recipe:** The final aggregated model combining these Ingredients to output a calibrated empirical probability that a prospect will trigger the same Event.

## What Each Recipe Must Have (5 Components)
For a recipe to exist in Analytics FC, all five components must be present:
1. **Key Indicator Ingredients:** Which developmental metrics carry the signal (e.g., overall, minutes).
2. **Canonical Length:** The precise developmental time window in years (e.g., exactly 4 years, ages 18 to 21).
3. **Canonical Indicator Movement Shape:** The aggregated movement pattern of the key indicators across all historical players that experienced the event.
4. **The Terminating Event:** The tangible, real-world career milestone the recipe leads to (e.g., winning a major 	eam_trophy or securing a major transfer).
5. **Aggregation Rule:** How multiple historical precedents are combined into one canonical signature to generate a final empirical probability score.

## Operating Rule
Every engineering, data, and mathematical decision must answer the question: 
> *Does this move us closer to instantiating the Correlation Cascade at production scale, with calibrated confidence, across all targeted football events?*
