# Data

All raw data is fetched via `make collect` and is not committed to this repository. Processed outputs are regenerated via `make pipeline`.

## Two committed files

`data/processed/final_model.json` — v1 model specification and benchmark comparison  
`data/processed/model_results_complete.csv` — full ablation table (24 model × feature set combinations)

## Forkable processed files (committed for exploration)

These files allow you to explore results without running the full pipeline:

| File | Rows | Description |
|------|-----:|-------------|
| `processed/fact_matchup_features.csv` | 49,479 | 51 pairwise features for all international matches |
| `processed/fact_matchup_archetype.csv` | 49,479 | 8 archetype labels per match |
| `processed/dim_team.csv` | 343 | Team dimension with confederation, region |
| `processed/dim_tournament.csv` | 24 | WC edition dimension |
| `processed/dim_stadium.csv` | 241 | Stadium dimension with city and coords |
| `processed/validation_2026.csv` | 73 | 2026 match predictions vs actual results |
| `processed/validation_summary.json` | — | 2026 validation headline metrics |
| `processed/predictions_2026.csv` | 73 | Pre-match probability predictions for 2026 |
| `processed/feature_importance.csv` | 32 | RF feature importances (best model) |
| `processed/cluster_eval.csv` | — | Unsupervised method comparison |

## Regenerating everything

```bash
make collect    # fetch raw data (~260MB Transfermarkt + Fjelstul)
make pipeline   # features → archetypes → models → validate
```

## Data sources and licenses

- **Fjelstul World Cup Database** — [github.com/jfjelstul/worldfootballR](https://github.com/jfjelstul/worldfootballR) — see source repository for license terms
- **International match results** — Mart Jürisoo dataset, public domain
- **Transfermarkt squad data** — scraped via `src/data_collection/collect_squads.py` — for research use only; see [transfermarkt.com](https://www.transfermarkt.com) terms of service
- **2026 live data** — Wikipedia squad pages + public match result feeds

> **Note on 2026 predictions:** `predictions_2026.csv` contains pre-match forecasts generated before each match using only information available at kickoff. The `validation_2026.csv` file compares these forecasts against actual results. These are research outputs, not production predictions.
