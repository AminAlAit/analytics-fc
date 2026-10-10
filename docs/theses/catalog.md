# Analytics FC: Master Theses Catalog

This catalog documents the core analytical and machine learning theses for Analytics FC, covering both player-level scouting models and club-level strategic intelligence.

---

## 1. Player Theses

### 1.1 The Superstar Ceiling Matcher (Flagship)
* **Thesis:** An 18 to 20 year-old prospect's micro-skill growth trajectory can be projected onto the prime physical and technical profiles (age 26 to 29) of all-time historical superstars.
* **Value:** Replaces binary trophy predictions with a continuous archetype projection, answering: "Which legendary player does this prospect develop into at their absolute peak?"
* **Methodology:** Unsupervised multi-dimensional similarity mapping across 34 granular attributes, combining cosine distance and Mahalanobis clustering.

### 1.2 The Moneyball Arbitrage Engine
* **Thesis:** Young players whose technical and physical profiles match high-overall benchmarks often trade at a 50%+ discount on transfer markets due to playing in lower leagues or lesser-known clubs.
* **Value:** Identifies market inefficiencies by discovering undervalued prospects before market correction.
* **Methodology:** Regression modeling predicting fair market value (`value_eur`) based on attribute vectors, ranking players by negative residual value.

### 1.3 Trait Elasticity and World-Class Catalysts
* **Thesis:** Not all skill attributes contribute equally to reaching elite status. Specific gateway traits (e.g. Vision, Reactions, Composure) act as mandatory filters for crossing from good (78-82) to world-class (88+).
* **Value:** Advises coaches and scouts on which trainable attributes deliver the highest marginal return on player development.
* **Methodology:** Feature attribution and interaction analysis using tree ensembles and SHAP values.

---

## 2. Club Theses

### 2.1 Squad Rebuild and Aging Cliff Alert
* **Thesis:** Clubs with a high proportion of minutes allocated to players aged 30+ in physically demanding roles experience sharp performance drop-offs within 2 seasons.
* **Value:** Flags impending squad crises early, recommending targeted renewal windows before competitive decline.
* **Methodology:** Time-series tracking of squad age distribution, physical decay slopes, and contract expiration horizons.

### 2.2 Wage Efficiency and Front-Office Overpay Index
* **Thesis:** High wage bills do not linearly correlate with on-pitch technical quality. Certain clubs systematically overpay for reputation over current performance.
* **Value:** Benchmarks front-office capital efficiency across European leagues.
* **Methodology:** Club-level regression analyzing weekly wage expenditure (`wage_eur`) against total squad technical capability.

### 2.3 Tactical DNA and System Compatibility
* **Thesis:** A player's individual success at a new club depends on how closely their skill vector aligns with the club's collective tactical style (e.g. High Press vs Direct Counter).
* **Value:** De-risks expensive transfers by scoring stylistic fit prior to acquisition.
* **Methodology:** Club centroid clustering compared against prospective player skill vectors.
