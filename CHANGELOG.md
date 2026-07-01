# Changelog

All notable changes to this project are documented here.

---

## [v1.0.0] — 2026-07-01 · Matchup Archetype Framework

### Added
- **8-archetype rule-based classification framework** covering all 1,037 WC matches (99.6% coverage 2014–2026): heavyweight clash, favorite vs. underdog, host pressure, generational transition, club power mismatch, tactical contrast, knockout volatility
- **Full Elo engine** fitted on 49,493 international matches 1872–2026 with K-factor scheduling and time-weighted decay
- **51 pairwise matchup features** across 3 sets: Elo-only (A), Elo + archetypes (B), full with squad (C)
- **26 contextual features**: rolling momentum (last-5 GD, streak), in-tournament stats, travel distance, venue altitude, WC qualifying form
- **KNN squad imputation** expanding usable full-feature training set from 415 → 677 matches
- **6-model-family ablation** (Logistic Regression, Random Forest, Gradient Boosting, MLP, HyperNEAT, GA Ensemble, GoL CA) × 3 feature sets via LOTO-CV over 23 World Cup editions
- **Best model (v1 freeze):** RF Optuna contextual — log-loss 0.9414, accuracy 59.0%
  - Params: `n_estimators=104, max_depth=6, min_samples_leaf=20, max_features=0.7`
  - Features: 32 (Elo + Archetypes + Squad-KNN + curated contextual)
- **2026 live validation pipeline** via GitHub Actions (every 6h): 53.4% accuracy, 73 matches
- **Augmentation experiment suite**: mirror, high-stakes proxy, rolling form, interaction features, stacking, calibration
- **Dixon-Coles bivariate Poisson model** (attack/defence per team, DC low-score correction)
- **Unsupervised clustering**: K-Means, Hierarchical, HDBSCAN, GMM, NMF across k=2–10
- **Documents**: proposal, 2 check-ins, full academic report

### Key negative findings (documented)
- Mirror augmentation hurts (Δ−0.023 log-loss): WC H/A encoding is real even at neutral venues
- Novel models (HyperNEAT, GoL CA) underperform classical RF at ~280 training samples/fold
- GoL CA search did not terminate on full feature set (exponential CA rule-table search complexity)
- Public-data ceiling: ~59% without bookmaker odds; Groll et al. 2019 reaches ~61% with odds

### Benchmark comparison
| Model | Accuracy |
|-------|:--------:|
| Majority baseline | 45.5% |
| Hvattum & Arntzen (2010) | 57.0% |
| **This project — v1** | **59.0%** |
| Groll et al. (2019) | 61.0% |
| Pinnacle closing market | 64.5% |

---

## Planned

### [v2.0.0] — Tournament-State Simulator
100K Monte Carlo branch simulations from any bracket state. Expected Tournament Value per team. Upset leverage analysis. Path dependency. Bracket-state visualizations.

### [v3.0.0] — In-Play Event Engine
Real-time win probability updates from match events. Possession/event slicing. Run-it-back replay. In-play calibration against v1 pre-match baseline.

### [v4.0.0] — Analytics Platform
REST API + dashboard. Model registry. Live market benchmarking. Stakeholder scenario planning.
