# FIFA World Cup 2026 — Matchup Archetype Classification

> **v1 — Matchup Archetype Framework** · July 2026

A research pipeline that classifies World Cup matchups into structural archetypes and tests whether those archetypes improve pre-match outcome prediction. Validated live against the 2026 FIFA World Cup.

## What it found

- **Archetype labels improve prediction** by +0.003–0.004 log-loss over Elo-only baselines across all model families tested
- **Best model (v1):** Random Forest + Optuna tuning on 32 features (Elo + Archetypes + Squad + contextual momentum) — **59.0% 3-class accuracy, log-loss 0.9414** via leave-one-tournament-out CV across 23 World Cup editions
- **2026 out-of-sample:** 53.4% accuracy on 73 live matches (+7.9pp above majority baseline), updated every 6 hours via GitHub Actions
- **Key negative finding:** Mirror augmentation hurts (Δ−0.023 log-loss) — home/away encoding carries real information even at neutral venues
- **Data ceiling:** Without bookmaker odds, ~59% is the public-data ceiling. Best published model with odds (Groll et al. 2019) reaches ~61%; Pinnacle market ~64.5%

## Reproduce

```bash
git clone https://github.com/sboettch/fifaworldcup2026-v1
cd fifaworldcup2026-v1
make install       # install dependencies
make collect       # fetch raw data from Fjelstul + Transfermarkt
make pipeline      # features → archetypes → models → validation
```

Requires Python 3.11+. Data fetching requires internet access; no API keys needed.

## Project structure

```
src/data_collection/    Raw data collectors (matches, squads, live 2026)
src/features/           Feature engineering (Elo, matchup features, archetypes, contextual)
src/models/             Supervised + unsupervised models, augmentation experiments, validation
docs/                   Proposal, check-ins, final report
outputs/figures/        Cluster PCA plots, feature importance, CV results
data/processed/         Final model spec + ablation table (all other CSVs regenerated)
```

## Documents

| Document | Description |
|----------|-------------|
| [docs/proposal.md](docs/proposal.md) | Original research proposal |
| [docs/checkin_1.md](docs/checkin_1.md) | Check-in 1: Data pipeline through archetype framework |
| [docs/checkin_2.md](docs/checkin_2.md) | Check-in 2: Supervised learning through v1 freeze |
| [docs/final_report.md](docs/final_report.md) | Full academic report |

## Version roadmap

| Version | Name | Focus |
|---------|------|-------|
| **v1** *(current)* | Matchup Archetype Framework | Data aggregation, archetypes, supervised/unsupervised learning, live validation |
| v2 | Tournament-State Simulator | 100K Monte Carlo branches, Expected Tournament Value, bracket-state visualizations |
| v3 | In-Play Event Engine | Real-time probability updates, possession/event slicing, in-play calibration |
| v4 | Analytics Platform | Dashboard, model registry, live monitoring, stakeholder workflows |

## Data sources

- **Match results:** [Fjelstul World Football Dataset](https://github.com/jfjelstul/worldfootballR) — 49,493 international matches 1872–2026
- **Squad data:** Transfermarkt via scraper (`src/data_collection/collect_squads.py`)
- **2026 live data:** Automated every 6h via GitHub Actions

## Citation

```
Boettcher, S. (2026). Classifying World Cup Matchup Archetypes: A Predictive Framework.
FIFA World Cup 2026 Research Project, v1.0.0.
https://github.com/sboettch/fifaworldcup2026-v1
```

## License

Code: [MIT License](LICENSE)
Documentation and findings: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Fjelstul dataset: see original repository for license terms.
