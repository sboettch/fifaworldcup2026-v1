# FIFA World Cup Longitudinal Dataset Strategy

## Working Goal

Build a longitudinal FIFA World Cup dataset that lets us classify matchups into meaningful archetypes: heavyweight clashes, favorite-underdog games, host-country pressure games, squad-continuity mismatches, tactical-style contrasts, and upset-risk matchups.

The first version should prioritize reliable historical coverage over beautiful but incomplete detail. We should build a stable tournament-match-team-player spine first, then enrich it with player context, advanced performance data, and pre-match strength signals.

## Pipeline Architecture

This project should be organized as a staged data pipeline, not a single scrape. The major boundary is between aggregation and harmonization.

| Stage | Purpose | Outputs | Key Rule |
| --- | --- | --- | --- |
| Stage 1A: Data aggregation | Pull and snapshot source data exactly as observed | Raw source files, source manifest, retrieval logs | Do not overwrite or silently clean raw data |
| Stage 1B: Data harmonization | Clean, normalize, join, deduplicate, impute, and validate | Staging tables, canonical dimensions, processed facts, QA reports | Every transformation must be reproducible and source-traceable |
| Stage 2: Analytical build | Engineer matchup features, explore patterns, discover archetypes, and test prediction targets | Feature matrix, EDA outputs, cluster labels, model baselines, visualizations | No future leakage; analysis uses frozen versioned data |

Working distinction:

- **Aggregation** answers: what data can we pull, from where, at what cadence, under what license, and with what coverage?
- **Harmonization** answers: how do those source rows become one usable longitudinal dataset with stable IDs, consistent schema, clean fields, imputed values where defensible, and explicit missingness flags?
- **Analysis** answers: what can the resulting dataset tell us about matchup archetypes, emergent clusters, and outcome prediction?

Stage 1 should produce both the data and the operational machinery for keeping it current. Stage 2 should not start from ad hoc notebooks against raw files; it should start from a frozen, versioned analytical dataset.

## Recommended Source Stack

### Dataset Attack Sheet

| Layer | Source | Grain | What We Mine | Main Join Keys | Main Risk |
| --- | --- | --- | --- | --- | --- |
| Historical backbone | Fjelstul World Cup Database | tournament, match, team-match, player-match, event | hosts, stadiums, squads, matches, results, player appearances, goals, bookings, substitutions, standings | `tournament_year`, source match/team/player IDs | coverage boundaries before 1970 for player appearance detail |
| Squad context | Wikipedia/FIFA squad pages | player-team-tournament | age, caps, goals, club, club country, coach, captain, shirt number | `year + team + shirt_number`, `year + team + player_name`, DOB | page/table inconsistency across older tournaments |
| Advanced performance | StatsBomb Open Data plus FBref/worldfootballR | modern match, team-match, player-match, event | xG, shots, passes, lineups, formations, event sequences, player/team style | `date + teams + score`, player name/team | modern-only coverage and source terms |
| Pre-match expectations | FiveThirtyEight SPI plus Elo/FIFA-style ratings | team-date, team-match | ratings, rank gaps, win/draw probabilities, projected goal difference, upset score | latest rating before kickoff, team mapping | incompatible rating systems across eras |

### 2026 Live Data Acquisition Overlay

The 2026 tournament needs a different ingestion mode from the historical tournaments. We should not wait for a post-tournament archival database update. Instead, we should create a provisional 2026 overlay that uses the same schema as the historical backbone, records source snapshots, and gets reconciled after the tournament.

As of June 29, 2026, the tournament is active/current data, so every 2026 row should carry freshness and confirmation metadata.

Recommended source ladder:

| Rank | Source | Use | Treatment |
| --- | --- | --- | --- |
| 1 | FIFA official tournament pages, match centre, match reports, and squad releases | Authoritative fixture, result, squad, lineup, match official, venue, and disciplinary facts | Canonical confirmation source; persist URL and retrieval timestamp |
| 2 | `openfootball/worldcup` 2026 files | Open seed for groups, venues, match schedule, scores, scorers, and knockout fixtures | Fast ingest source; verify against FIFA before marking final |
| 3 | 2026 Wikipedia squad and tournament pages | Structured squad tables, coaches, clubs, caps/goals, group context, late updates | Useful parser input; validate against FIFA or national associations |
| 4 | FBref/worldfootballR and similar stat providers | Player/team box stats and advanced modern performance features as they become available | Enrichment layer only; never the canonical result source |
| 5 | Our own Elo rebuild, plus SPI if available | Pre-match team strength and expectations | Rating snapshots must be frozen before kickoff |

2026-specific fields to add:

- `data_status`: `scheduled`, `provisional`, `confirmed`, `archived`
- `source_priority`: numeric source rank from the ladder above
- `source_url`
- `source_retrieved_at`
- `source_last_modified_at`, where available
- `match_status`: `scheduled`, `in_progress`, `final`, `final_after_extra_time`, `final_after_penalties`
- `lineup_status`: `missing`, `predicted`, `official`
- `result_version`
- `needs_reconciliation_flag`

2026 ingestion cadence:

1. Pull official FIFA pages and openfootball 2026 files daily during the tournament.
2. During matchdays, run a post-match refresh after each final whistle window.
3. Save immutable raw snapshots by retrieval date in `data/raw/2026_live/YYYY-MM-DD/`.
4. Parse into staging tables first: `stg_2026_matches`, `stg_2026_squads`, `stg_2026_events`, and `stg_2026_venues`.
5. Promote rows to canonical processed tables only after source priority and row-level QA checks pass.
6. Freeze pre-match ratings before kickoff; never recompute them using post-match information.
7. After the tournament, reconcile the provisional overlay against Fjelstul, FIFA technical reports, and final stat-provider exports.

2026 QA checks:

- Every completed match has two team-match rows, one final score, and one match-status value.
- Host city, venue, timezone, and kickoff time reconcile across FIFA and openfootball.
- Goal events reconcile to final score, including own goals and penalties.
- Knockout matches carry extra-time and penalty-shootout fields when relevant.
- Squad rows have official roster status, late replacement flags, and a source timestamp.
- Ratings used for pre-match expectations are timestamped before kickoff.

### 2026 Automation Operating Model

Because the 2026 tournament is live, we should automate both aggregation and harmonization. The goal is not only to avoid manual reminders; it is to create a reliable audit trail for changing data.

Recommended first implementation:

1. Use a hosted scheduled workflow, preferably GitHub Actions `schedule`, to run the pipeline on the default branch. GitHub scheduled workflows use cron syntax, run in UTC by default, and can run as frequently as every five minutes. For our use case, a daily run plus matchday post-match runs is enough.
2. Keep a manual dispatch trigger too, so we can force a refresh after a major match, roster correction, or source-format change.
3. Run the same sequence every time:
   - `collect_2026_sources.py`
   - `parse_2026_sources.py`
   - `harmonize_worldcup.py --tournament 2026`
   - `validate_worldcup.py --tournament 2026`
   - `publish_dataset.py --target drive_or_release`
4. Publish versioned outputs to a stable access layer: Google Drive, Google Cloud Storage, BigQuery, GitHub Releases, or another shared dataset location.
5. Save run logs and QA reports with each dataset version so downstream analysis can cite exactly which data snapshot it used.

Recommended first cadence:

- Daily during the tournament for schedules, squads, venues, standings, and source drift.
- Post-match on matchdays after final-whistle windows.
- Extra manual refresh when FIFA, national associations, or major stat providers issue corrections.
- Final reconciliation pass after the tournament ends.

Hosted automation options:

| Option | Best Use | Tradeoff |
| --- | --- | --- |
| GitHub Actions scheduled workflow | Fastest first version if the code lives in GitHub | Great for repo-native automation, but long-term storage and Drive publishing need secrets |
| Google Cloud Scheduler + Cloud Run or Cloud Functions | More production-like hosted pipeline, especially if publishing to Google storage or Drive | More setup, service accounts, and cloud configuration |
| Local cron | Development fallback only | Fragile because it depends on one machine being awake and configured |

Drive/publication approach:

- Store raw and processed outputs locally in the repo structure during each run.
- Upload the latest release bundle to Google Drive or another shared destination through an authenticated service account.
- Include `manifest.json`, `source_manifest.csv`, `qa_report.html` or `qa_report.md`, and versioned `parquet`/`csv` outputs.
- Never make Drive the only copy of raw source evidence; Drive should be a publication layer, while raw snapshots remain versioned in the data pipeline storage.

### 1. Historical World Cup Backbone

**Primary source:** Fjelstul World Cup Database  
**URL:** https://github.com/jfjelstul/worldcup

**Role in our dataset:** This should be the canonical backbone. It already contains linked tables for tournaments, host countries, stadiums, matches, teams, squads, player appearances, goals, bookings, substitutions, penalty kicks, and standings.

**Coverage:** Men's World Cups from 1930 through 2022, plus women's World Cup data. For our first pass, filter to men's tournaments only.

**Tables to ingest first:**

- `tournaments`
- `host_countries`
- `stadiums`
- `matches`
- `teams`
- `qualified_teams`
- `squads`
- `team_appearances`
- `player_appearances`
- `goals`
- `bookings`
- `substitutions`
- `penalty_kicks`
- `tournament_standings`
- `group_standings`

**Why this is optimal:** It gives us the relational spine we need without scraping every historical tournament page from scratch. It has stable IDs and is available in CSV, JSON, RData, and SQLite formats.

**Mining approach:**

1. Pull the CSV or SQLite release into `data/raw/fjelstul_worldcup/`.
2. Filter to men's tournaments.
3. Preserve source IDs as `source_*_id` columns.
4. Create canonical internal IDs for tournaments, teams, players, stadiums, and matches.
5. Build the first processed tables:
   - `dim_tournament`
   - `dim_team`
   - `dim_player`
   - `dim_stadium`
   - `fact_match`
   - `fact_team_match`
   - `fact_player_match`
   - `fact_event`

**Known limitations:** Player appearances only become more detailed from 1970 onward because FIFA match reports do not consistently support substitutions before then. We should explicitly mark this as a coverage boundary instead of imputing false precision.

### 2. Squad and Player Context Enrichment

**Primary source:** Wikipedia tournament squad pages, validated against FIFA squad releases where needed  
**Example URLs:**

- https://en.wikipedia.org/wiki/1930_FIFA_World_Cup_squads
- https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads
- https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads

**Role in our dataset:** Enrich the Fjelstul `squads` table with player-level context at the start of each tournament.

**Fields to mine:**

- `tournament_year`
- `team`
- `group`
- `shirt_number`
- `position`
- `player_name`
- `date_of_birth`
- `age_at_tournament_start`
- `pre_tournament_caps`
- `pre_tournament_goals`
- `club`
- `club_country`
- `coach`
- `captain_flag`

**Why this is optimal:** These pages expose the composition of each country's team in a structured table format and include exactly the context we need for squad archetypes: age, caps, goals, club affiliation, and coach. Fjelstul gives us the spine; this layer gives us richer player and club context.

**Mining approach:**

1. Generate URLs from the year pattern: `{YEAR}_FIFA_World_Cup_squads`.
2. Use structured HTML table extraction, not manual copy/paste.
3. Normalize table headers across eras. Older tournaments may omit goals or use different coach labels.
4. Join to Fjelstul squads using a tiered match strategy:
   - exact `tournament_year + team + shirt_number`
   - exact `tournament_year + team + normalized_player_name`
   - fuzzy `player_name + date_of_birth`
5. Flag unresolved joins for manual review.

**Known limitations:** Wikipedia is editable and historical pages vary in structure. We should snapshot scraped raw HTML/CSV outputs and log scrape dates. For high-impact rows, use FIFA technical reports or official squad PDFs as validation.

### 3. Advanced Match and Player Performance Layer

**Primary sources:**

- StatsBomb Open Data: https://github.com/statsbomb/open-data
- worldfootballR FBref tools: https://jaseziv.github.io/worldfootballR/reference/index.html

**Role in our dataset:** Create high-resolution player and team performance features for the modern era. This layer should not be forced across the full 1930-2022 history.

**Fields to mine or derive:**

- Player minutes, starts, substitutions
- Shots, goals, assists, expected goals where available
- Passes, carries, pressures, defensive actions where available
- Lineups and formations
- Team possession and shot profiles
- Match event sequences
- Penalty shootout details

**Why this is optimal:** StatsBomb provides free event data with JSON files for competitions, matches, events, lineups, and selected 360 data. FBref, accessed carefully through tooling such as worldfootballR, can add player and team season-style box stats where coverage exists.

**Mining approach:**

1. Start with StatsBomb because the data is structured JSON and openly documented.
2. Filter competitions to FIFA World Cup seasons available in the open-data release.
3. Aggregate raw events to:
   - `fact_modern_player_match`
   - `fact_modern_team_match`
   - `fact_modern_match_style`
4. Join matches to the backbone using `date + team names + competition + score`.
5. Join players using `team + player_name`, then improve with date of birth or external player IDs where available.
6. Treat FBref/worldfootballR as an enrichment path, not the canonical source, because coverage and scrape stability can change.

**Known limitations:** This is modern-era only. It is excellent for tactical and player-style archetypes in recent tournaments, but it should not define the historical archetype schema by itself.

### 4. Pre-Match Team Strength and Expectation Layer

**Primary sources:**

- FiveThirtyEight Soccer SPI data: https://github.com/fivethirtyeight/data/tree/master/soccer-spi
- World Football Elo / FIFA ranking methodology references:
  - https://en.wikipedia.org/wiki/World_Football_Elo_Ratings
  - https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking

**Role in our dataset:** Quantify whether a matchup was expected to be close, lopsided, volatile, or upset-prone before the match began.

**Fields to mine or derive:**

- `team_rating_pre_match`
- `opponent_rating_pre_match`
- `rating_gap`
- `rating_percentile`
- `rank_gap`
- `win_probability`
- `draw_probability`
- `projected_goal_diff`
- `upset_result_flag`
- `surprise_score`

**Why this is optimal:** Matchup archetypes need expectation, not just outcome. A 1-0 match means different things when it is Brazil vs Germany, Brazil vs a debutant, or two evenly rated mid-tier teams. SPI gives recent pre-match forecasts; Elo/FIFA-style ratings can backfill longer historical context.

**Mining approach:**

1. Ingest FiveThirtyEight `spi_matches_intl.csv` and `spi_global_rankings_intl.csv` for modern international coverage.
2. For pre-2016 World Cups, build or ingest a historical Elo layer from international match results.
3. Store ratings as date-effective snapshots, not tournament-level constants.
4. Join to `fact_team_match` using the latest rating available before match kickoff.
5. Calculate rating gaps symmetrically so matchups are comparable regardless of home/away orientation.

**Known limitations:** SPI coverage begins in the modern era. FIFA rankings begin in 1992 and the methodology changed over time. For full history, a reproducible Elo rebuild from international results may be cleaner than mixing incompatible rating systems without flags.

## Core Longitudinal Model

The core model should support four grains:

- **Tournament-team grain:** one row per team per World Cup.
- **Match grain:** one row per match.
- **Team-match grain:** two rows per match, one for each team.
- **Player-match grain:** one row per player appearance per match where coverage allows.

Recommended processed tables:

| Table | Grain | Purpose |
| --- | --- | --- |
| `dim_tournament` | tournament | Year, host country/countries, dates, format |
| `dim_team` | team | Canonical country/team IDs, confederation, ISO codes |
| `dim_player` | player | Canonical player IDs, birth date, nationality flags where available |
| `dim_stadium` | stadium | Stadium, city, country, capacity |
| `bridge_squad_player_tournament` | player-team-tournament | Roster membership and tournament-start context |
| `fact_match` | match | Date, stage, stadium, teams, score, extra time, penalties |
| `fact_team_match` | team-match | Goals for/against, result, opponent, host indicator, rest, travel, rating context |
| `fact_player_match` | player-match | Starts, minutes, goals, cards, substitutions, advanced modern stats |
| `fact_event` | event | Goals, cards, substitutions, penalties, and modern event data |
| `fact_matchup_archetype` | match | Derived matchup labels and feature scores |

## Join Strategy

Use Fjelstul IDs wherever possible, then layer external sources onto those IDs.

Canonical keys:

- `tournament_id`
- `tournament_year`
- `match_id`
- `team_id`
- `opponent_team_id`
- `player_id`
- `stadium_id`

External-source mapping tables:

- `map_team_names`
- `map_player_names`
- `map_stadium_names`
- `map_competition_ids`
- `map_source_match_ids`

Name normalization rules:

- Store original names and normalized names.
- Normalize punctuation, accents, and federation-era country names separately.
- Do not collapse historically distinct entities without explicit mapping rules. Examples: Germany/West Germany/East Germany, Russia/Soviet Union, Serbia/Yugoslavia, Czech Republic/Czechoslovakia.
- Keep a `historical_team_entity_id` and a `modern_country_id` if we want both historical accuracy and modern country grouping.

## Initial Matchup Archetypes

These should begin as transparent rule-based labels, then become model-assisted clusters after we have enough features.

| Archetype | Definition Sketch | Required Features |
| --- | --- | --- |
| Heavyweight clash | Both teams in top rating percentile or both recent semifinal/final-level teams | rating percentile, prior tournament finish, Elo/SPI |
| Favorite vs underdog | Rating gap above threshold before kickoff | pre-match rating gap, win probability |
| Host pressure game | Host nation involved, or opponent playing host on home continent | host flag, venue country, confederation |
| Generational transition | One or both squads have unusually young/old age profile or low returning experience | squad age, caps, prior appearances |
| Club-power mismatch | One squad has much higher top-league or elite-club representation | club country, club tier, player market/club proxy |
| Tactical contrast | Teams differ sharply in possession, directness, pressing, shot profile | modern event/FBref/StatsBomb data |
| Knockout volatility | Elimination match with high parity or penalty/extra-time likelihood | stage, rating gap, knockout flag, extra-time history |
| Upset realized | Underdog wins or advances against expectation | rating gap, result, penalties, surprise score |

## Exploratory Analysis and Modeling Roadmap

Once Stage 1 produces a stable, versioned analytical dataset, the next phase should ask what structure emerges from the data before forcing labels onto it.

### Stage 2A: Exploratory Data Analysis

Core questions:

- How do rating gaps, host status, squad age, squad experience, club composition, rest, travel, and stage relate to outcomes?
- Which features are stable across eras, and which only make sense in the modern advanced-data period?
- Which features distinguish close matches, blowouts, upsets, extra-time matches, and penalty shootouts?
- Are there eras or tournaments where the data-generating process changes enough to require separate models or flags?

Recommended outputs:

- Missingness and coverage maps by year, source, feature, team, and tournament stage.
- Distribution plots for ratings, age/caps, experience, host flags, goal difference, shot/xG features, and upset indicators.
- Correlation heatmaps, but separated by feature family so modern-only variables do not dominate the full-history story.
- Matchup scorecards that show the two teams side by side before outcome variables are revealed.

### Stage 2B: Emergent Cluster Discovery

The first matchup archetypes should be transparent rules, but we should also search for emergent clusters.

Recommended approach:

1. Build a feature matrix at match grain and team-match grain.
2. Split features into pre-match-only, outcome-aware, and modern-performance-only groups.
3. Standardize numeric fields and one-hot or target-encode categorical fields carefully.
4. Use dimensionality reduction for exploration: PCA for interpretability, UMAP for nonlinear visual structure.
5. Compare clustering methods: k-means, hierarchical clustering, Gaussian mixture models, and HDBSCAN.
6. Evaluate stability across random seeds, tournaments, eras, and feature families.
7. Translate stable clusters into human-readable archetype labels.

Potential cluster outputs:

- `cluster_id`
- `cluster_label`
- `cluster_confidence`
- `feature_family`
- `era_applicability`
- `nearest_archetype_rule`
- `cluster_stability_score`

### Stage 2C: Outcome Prediction

Prediction should come after the aggregation and harmonization layer is trustworthy. The first target does not need to be perfect; the main goal is to learn whether matchup features carry signal.

Candidate targets:

- Win/draw/loss before kickoff.
- Goal difference or expected margin.
- Upset probability.
- Knockout advancement.
- Extra-time or penalty-shootout likelihood.
- Goal-total band, such as low-scoring versus high-scoring match.

Modeling guardrails:

- Separate pre-match prediction features from post-match explanation features.
- Use time-aware validation, such as train on earlier tournaments and test on later tournaments.
- Include a simple baseline first: rating-gap-only logistic regression or Elo-only expected outcome.
- Add richer models only after the baseline is clear: regularized logistic regression, random forest, gradient boosting, and calibrated probability models.
- Evaluate calibration, not just accuracy, because matchup archetyping needs trustworthy probabilities.
- Report interpretable feature effects so predictions can feed back into archetype definitions.

Visualization ideas:

- Archetype map: matches positioned by rating gap, host advantage, squad experience gap, and style distance.
- Tournament timeline: how matchup archetypes appear by stage and year.
- Upset matrix: expected strength versus actual result.
- Host-path view: how host-country matches differ from neutral matches.
- Cluster profile cards: representative matches, defining features, and outcome tendencies for each archetype.

## Strategy of Attack

### Phase 0: Project Setup

Create the source registry, folder structure, and run manifest before mining anything.

Deliverables:

- `data/source_manifest.csv`
- `data/run_manifest.csv`
- `docs/data_dictionary.md`
- `docs/source_notes.md`
- `notebooks/00_source_audit.ipynb` or equivalent script

Manifest fields:

- `source_name`
- `source_url`
- `license`
- `coverage_years`
- `grain`
- `raw_path`
- `ingested_at`
- `pipeline_run_id`
- `known_limitations`

### Phase 1A: Data Aggregation

Mine and snapshot raw sources without silently changing them.

Deliverables:

- Raw Fjelstul snapshot in `data/raw/fjelstul_worldcup/`.
- Raw squad-page snapshots by year.
- Raw StatsBomb, FBref/worldfootballR, SPI, and rating-source snapshots where available.
- Raw 2026 live snapshots under `data/raw/2026_live/YYYY-MM-DD/`.
- Hosted scheduled workflow for 2026 live ingestion.

QA checks:

- Every source file has a source manifest row.
- Every live 2026 pull has a retrieval timestamp, source URL, and pipeline run ID.
- Aggregation scripts can be rerun without overwriting historical raw evidence.

### Phase 1B: Data Harmonization

Convert raw and staging data into a clean longitudinal schema.

Deliverables:

- Canonical dimensions: `dim_tournament`, `dim_team`, `dim_player`, `dim_stadium`.
- Canonical facts: `fact_match`, `fact_team_match`, `fact_player_match`, `fact_event`.
- External-source mapping tables for teams, players, matches, stadiums, and competitions.
- Imputation log explaining every filled value and every value intentionally left missing.
- Row-count and referential-integrity QA reports.

QA checks:

- Every match has exactly two team-match rows.
- Score fields reconcile between match, team-match, and event tables.
- Host-country flags reconcile with the tournament host table.
- Squad sizes align with tournament rules for the year.
- Rating timestamps are before kickoff.
- Modern advanced-stat fields are coverage-flagged so unavailable history is not treated as zero.

### Phase 1C: Analytical Dataset Build

Freeze a versioned feature matrix that analysts and models can use.

Deliverables:

- `analysis_match_features`
- `analysis_team_match_features`
- `analysis_player_match_features`, where coverage allows
- `feature_dictionary.md`
- Versioned release bundle in `data/releases/`

Required feature families:

- Pre-match strength: rating gap, win probability, rank gap, upset expectation.
- Host and venue context: host flag, continent, city, stadium, timezone, travel/rest later.
- Squad context: age, caps, goals, club footprint, coach, returning experience.
- Match context: stage, knockout flag, group standing pressure, extra-time/penalty availability.
- Modern performance: xG, shot profile, passing, possession, pressing, and event-derived style features where available.

### Phase 2A: Exploratory Data Analysis

Use the frozen analytical dataset to understand structure before modeling.

Deliverables:

- Missingness and coverage maps.
- Correlation and distribution reports.
- Era-by-era feature diagnostics.
- Matchup scorecards for representative games.
- Initial hypothesis list for archetype definitions.

### Phase 2B: Archetype Discovery

Start with transparent rules, then compare against emergent clusters.

Deliverables:

- `fact_matchup_archetype`
- `docs/archetype_definitions.md`
- Cluster notebooks or scripts for PCA/UMAP plus k-means, hierarchical clustering, Gaussian mixtures, and HDBSCAN.
- Cluster stability report.

First-pass archetype features:

- `rating_gap`
- `both_top_quartile_flag`
- `host_involved_flag`
- `knockout_flag`
- `squad_avg_age_gap`
- `squad_caps_gap`
- `top_league_player_share_gap`
- `prior_world_cup_experience_gap`
- `style_distance_modern_only`
- `upset_flag`

### Phase 2C: Outcome Prediction

Treat prediction as a downstream test of signal, not the first product.

Deliverables:

- Baseline rating-only model.
- Pre-match-only model for win/draw/loss, goal difference, upset probability, and advancement where relevant.
- Temporal validation report.
- Calibration plots and feature-importance summary.
- Prediction-to-archetype interpretation notes.

Guardrails:

- No post-match variables in pre-match prediction targets.
- Train/test splits should respect time and tournament boundaries.
- Prediction outputs should include probability calibration, not just class accuracy.

### Phase 3: Future Directions

After the core dataset is stable, add higher-ambiguity layers:

- Sentiment: press coverage, fan discourse, broadcast narratives.
- Logistics: travel distance, rest days, time zones, climate, altitude.
- Injuries and availability: pre-tournament injury reports, suspensions.
- Economics: player market values, club wage tiers, domestic league strength.
- Coaching continuity: manager tenure, style history, tactical lineage.

## First Version Definition of Done

Version 0.1 is done when we have:

- A reproducible ingest of the Fjelstul backbone.
- A scheduled 2026 live overlay that runs aggregation, harmonization, QA, and publication without manual reminders.
- Raw snapshots, source-priority flags, and reconciliation metadata for every 2026 pipeline run.
- A roster enrichment table for all men's World Cups with join-quality flags.
- A pre-match rating/expectation layer for at least modern tournaments, with a path to historical Elo.
- A modern advanced-performance layer for the tournaments available in StatsBomb/FBref.
- A frozen analytical feature matrix for EDA, clustering, and prediction experiments.
- A match-level archetype table with transparent labels and no silent missingness.
- A source manifest and QA report that explain every known coverage boundary.

## Immediate Next Step

Create the source registry and pipeline skeleton first, then ingest the Fjelstul CSV or SQLite data and wire the 2026 scheduled overlay. That gives us both the historical backbone and the live-data machinery before downstream analysis begins.

## Sources

- Fjelstul World Cup Database: https://github.com/jfjelstul/worldcup
- FIFA World Cup 26 official tournament pages: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
- openfootball World Cup 2026 files: https://github.com/openfootball/worldcup/tree/master/2026--usa
- StatsBomb Open Data: https://github.com/statsbomb/open-data
- worldfootballR reference: https://jaseziv.github.io/worldfootballR/reference/index.html
- FiveThirtyEight Soccer SPI data: https://github.com/fivethirtyeight/data/tree/master/soccer-spi
- GitHub Actions scheduled workflows: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
- Google Cloud Scheduler documentation: https://cloud.google.com/scheduler/docs
- Google Drive API upload documentation: https://developers.google.com/workspace/drive/api/guides/manage-uploads
