# `src/models/experimental/`

These scripts were part of the v1 investigation into whether novel model architectures could outperform classical Random Forest on the World Cup matchup prediction task. They are included here to document the findings transparently — including the negative ones.

## Results summary

| Script | Model | Best result | Verdict |
|--------|-------|:-----------:|---------|
| `run_ablation.py` | Full 6-family ablation runner | — | Use to reproduce Table 3 in `docs/final_report.md` |
| `augment_and_improve.py` | Mirror aug, stacking, calibration | Δ−0.023 (mirror hurts) | Key negative finding |
| `ga_ensemble.py` | GA-evolved archetype-conditioned ensemble | ll=0.9831 on C | Doesn't beat RF |
| `hyperneat_classifier.py` | HyperNEAT CPPN neuroevolution | ll=1.0717 on A | Doesn't beat RF |
| `gol_classifier.py` | Game-of-Life CA rule-table classifier | ll=1.2054 on A; **DNF on C** | CA search scales exponentially |
| `dixon_coles.py` | Dixon-Coles (1997) bivariate Poisson | Not completed (LOTO intractable locally) | Foundation for v2 |

## Why they don't beat Random Forest

The effective training set per LOTO fold is ~280 matches. Neural/evolutionary methods need substantially more data to express their capacity advantage. Random Forest's inductive bias (low-depth, high-regularisation) is well-matched to small tabular datasets with correlated features. This is a finding about the problem, not a failure of the implementations.

## Why they're still here

- The GoL DNF is informative: CA rule-table search scales exponentially with feature dimensionality, which has implications for any future attempt to use CA-based methods on richer feature sets.
- `augment_and_improve.py` documents why mirror augmentation doesn't work (H/A encoding is real even at neutral venues) — a result worth preserving.
- `dixon_coles.py` is the right approach for v2 (Tournament-State Simulator), where fitting team attack/defence parameters at tournament start rather than per LOTO fold makes it tractable.

## These scripts are NOT part of `make pipeline`

Run them individually if you want to reproduce the ablation or explore the experimental models:
```bash
python3 -m src.models.experimental.run_ablation
python3 -m src.models.experimental.augment_and_improve
```
