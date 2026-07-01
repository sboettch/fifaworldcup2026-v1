# FIFA World Cup 2026 — Matchup Archetype Classification

> **v1 — Matchup Archetype Framework** · July 2026 · by [sophia](mailto:sboettcher@ub313.net)

A research pipeline that classifies World Cup matchups into structural archetypes and tests whether those archetypes improve pre-match outcome prediction. Validated live against the 2026 FIFA World Cup.

Questions or feedback → [sboettcher@ub313.net](mailto:sboettcher@ub313.net)

---

## What it found

- **Archetype labels improve prediction** by +0.003–0.004 log-loss over Elo-only baselines across model families tested
- **Best model (v1):** Random Forest + Optuna tuning on 32 features (Elo + Archetypes + KNN-imputed Squad + contextual momentum) — **59.0% 3-class accuracy, log-loss 0.9414** via leave-one-tournament-out CV across 23 World Cup editions
- **2026 out-of-sample:** 53.4% accuracy on 73 live matches (+7.9pp above majority baseline), updated throughout the tournament via GitHub Actions
- **Key negative finding:** Mirror augmentation hurts (Δ−0.023 log-loss) — home/away encoding carries real information even at neutral venues. KNN imputation of missing squad data beats hyperparameter tuning at this data scale.
- **Data ceiling:** Without bookmaker odds, ~59% is the public-data ceiling. Best published model with odds (Groll et al. 2019) reaches ~61%; Pinnacle market ~64.5%

| Benchmark | Accuracy |
|-----------|:--------:|
| Majority baseline | 45.5% |
| Hvattum & Arntzen (2010) Elo-only | 57.0% |
| **This project — v1** | **59.0%** |
| Groll et al. (2019) with bookmaker odds | 61.0% |
| Pinnacle closing market | 64.5% |

---

## Reproduce

```bash
git clone https://github.com/sboettch/fifaworldcup2026-v1
cd fifaworldcup2026-v1
make install       # install dependencies
make collect       # fetch raw data from Fjelstul + Transfermarkt
make pipeline      # features → archetypes → models → validate
```

Requires Python 3.11+. No API keys needed.

Or explore without running anything — the key processed files are already committed under `data/processed/`. See [`data/README.md`](data/README.md).

---

## Project structure

```
src/
  data_collection/   Raw data collectors (matches, squads, live 2026)
  features/          Feature engineering (Elo, matchup features, archetypes, contextual)
  models/            Supervised + unsupervised models, validation
  models/experimental/  Novel architectures and augmentation experiments (documented negative findings)
docs/                Proposal, check-ins, final report
data/processed/      Forkable results + model spec (large tables regenerated via pipeline)
outputs/figures/     Cluster PCA plots, feature importance, CV log-loss
```

---

## Documents

| Document | Description |
|----------|-------------|
| [docs/proposal.md](docs/proposal.md) | Research proposal — framing, hypotheses, methods |
| [docs/checkin_1.md](docs/checkin_1.md) | Check-in 1: data pipeline through archetype framework |
| [docs/checkin_2.md](docs/checkin_2.md) | Check-in 2: supervised learning through v1 freeze |
| [docs/final_report.md](docs/final_report.md) | Full report — methods, results, discussion |

---

## Version roadmap

| Version | Name | What it adds |
|---------|------|-------------|
| **v1** *(this)* | Matchup Archetype Framework | Data pipeline, archetypes, supervised/unsupervised models, live validation |
| v2 | Tournament-State Simulator | 100K Monte Carlo branches, Expected Tournament Value, bracket-state visualizations |
| v3 | In-Play Event Engine | Real-time probability updates from match events, in-play calibration |
| v4 | Analytics Platform | Dashboard, API, live monitoring, scenario planning |

---

## Data sources

- **Match results:** [Fjelstul World Football Dataset](https://github.com/jfjelstul/worldfootballR) — 49,493 international matches 1872–2026
- **Squad data:** Transfermarkt via `src/data_collection/collect_squads.py` — research use only
- **2026 live data:** Wikipedia squad pages + public match result feeds, automated via GitHub Actions

---

## Citation

```
sophia (2026). FIFA World Cup 2026 — Matchup Archetype Classification, v1.0.0.
https://github.com/sboettch/fifaworldcup2026-v1
```

## License

Code: [MIT](LICENSE) · Docs & findings: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
