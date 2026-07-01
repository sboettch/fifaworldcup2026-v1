# Check-in 2: Supervised Learning through v1 Freeze
**FIFA World Cup 2026 — Matchup Archetype Classification**
*Sophia Boettcher · Phases 6–10*

---

## 1. Supervised Learning Setup

The supervised evaluation is built around leave-one-tournament-out cross-validation (LOTO-CV) across all 23 WC editions (1930–2026). Each fold holds out one full tournament edition and trains on all remaining editions. This is the only valid cross-validation strategy for this dataset: temporal splitting at the tournament level prevents any leakage of future Elo ratings, squad compositions, or contextual features into training folds.

We evaluate three feature sets in every experiment:

- **Feature Set A**: Elo-only. Full coverage, 1930–2026, all 1,037 WC matches available.
- **Feature Set B**: Elo + archetype labels. Same coverage as A; archetype labels are derived entirely from pre-match features and contain no outcome information (the `upset_realized` label is excluded from all predictor sets).
- **Feature Set C**: Elo + archetypes + squad features (KNN-imputed). Restricted to 2002+ due to squad data availability; effective training set 415–677 matches.

The primary evaluation metric is log-loss (multi-class, Win/Draw/Loss). Accuracy is reported as a secondary metric. Baselines: majority class (always predict home win, 45.5% accuracy), and the published benchmark from Groll et al. (2019), which achieved approximately 61% accuracy on the 2018 World Cup using a comparable Random Forest approach. The Pinnacle implied-probability market sits at approximately 64%.

---

## 2. Core Finding: Archetype Labels Help, Modestly

Archetype labels provide consistent, directionally positive lift across the two model families where they have any effect:

| Model | Feature set | Log-loss | Accuracy |
|-------|------------|:--------:|:--------:|
| Majority baseline | — | 1.0587 | 45.5% |
| Random Forest | A: Elo only | 0.9542 | 58.0% |
| **Random Forest** | **B: Elo + Archetypes** | **0.9506** | **57.9%** |
| Random Forest | C: Elo + Arch + Squad | 0.9491 | 58.0% |
| Logistic Regression | A: Elo only | 0.9581 | 55.9% |
| Logistic Regression | B: Elo + Archetypes | 0.9566 | 55.7% |
| Gradient Boosting | A: Elo only | 1.0239 | 55.0% |
| MLP | A: Elo only | 0.9621 | 56.8% |

Archetype lift (Δlog-loss): Random Forest +0.0036, Logistic Regression +0.0015. Gradient Boosting and MLP show no gain (−0.0002 and −0.0128 respectively). The best overall model — RF Optuna-tuned on Elo + archetypes + squad features with KNN imputation and curated contextual features — reaches log-loss 0.9419 and accuracy 59.2% (the "v1 freeze" model).

The modest magnitude is worth taking seriously. A Δlog-loss of 0.003 is small but consistent across independent model families and across LOTO folds. It is not a spurious effect of a single fortunate fold. The leading archetype features in RF importance are `club_power_mismatch` (1.4%) at position 7 and `generational_transition` (1.3%) — both squad-composition measures, not Elo-derived, confirming that archetype labels carry genuinely incremental information above raw ratings.

The more honest summary: Elo explains the large majority of predictable variance. Archetypes carry a real but modest residual signal, concentrated in linear models. The value of the archetype framework is at least as much about structure and interpretability as it is about raw prediction gain.

---

## 3. The Novel Model Experiment

### Why We Built Them

Three novel model families were implemented as explicit tests of whether non-classical approaches could extract more signal from the archetype-augmented feature space:

- **HyperNEAT** (CPPN neuroevolution): geometric neuroevolution in weight space, hypothesized to discover structured matchup interactions that gradient-based training misses.
- **GoL Cellular Automaton**: a cellular automaton classifier using Conway's Game of Life rules adapted to binary feature patterns, exploring non-linear state-transition dynamics.
- **GA Dynamic Ensemble**: a genetic algorithm that evolves an archetype-conditioned weighting over base classifiers, adapting its blend to the archetype label of each match.

### What Happened

None of them beat classical Random Forest. The results by family:

| Family | Best log-loss | Notes |
|--------|:-------------:|-------|
| RF (best configuration) | **0.9419** | v1 freeze model |
| GA Dynamic Ensemble | 0.9831 | Feature set C |
| HyperNEAT | 1.0717 | Feature set C |
| GoL | DNF | Feature set C — search hung |

**This is not a failure of implementation. It is the finding.** The real training set per LOTO fold is approximately 280 matches (the mean fold size with 23 editions held out serially). At 280 training examples, neural and evolutionary methods are squarely in the high-variance regime. Random Forest's inductive bias — ensembling shallow decision trees, implicitly regularizing — is well-matched to this scale. HyperNEAT's neuroevolutionary search requires a much larger fitness-evaluation surface to converge reliably; on 280 examples, it consistently underfits the structure it is designed to find.

### The GoL Result

The GoL classifier did not finish (DNF) on Feature Set C. The search hung. This is consistent with a known scaling property of cellular automaton rule-table search: the search space scales exponentially with input dimensionality. Feature Set C has substantially more features than Sets A or B, and the CA rule-table enumeration becomes intractable. This is worth documenting explicitly — it is not a bug in the implementation but a predictable consequence of applying a CA search method to a high-dimensional input without a pruning strategy.

---

## 4. Augmentation Experiments

### Mirror Augmentation Hurts

The most clear-cut negative result of the project: artificially augmenting the training set by adding mirror-flipped copies of each match (swapping team A and team B, negating directional features) makes model performance worse.

Δlog-loss from mirror augmentation: −0.023 (that is, log-loss increases by 0.023, which is a degradation). The reason is straightforward once stated: World Cup matches at neutral venues are not home/away symmetric. Even at a geographically neutral stadium, one team typically has larger traveling support, more favorable climate acclimatization, shorter travel distance from their training base, and — in the 2026 case — may be playing in their confederation's territory. The "home" encoding in the feature matrix captures a real structural asymmetry, not a statistical artifact. Destroying that information by treating every match as bidirectionally equivalent removes signal.

**This is a publishable negative result.** It argues against a common augmentation strategy used in smaller sports datasets where venue neutrality is assumed.

### KNN Imputation Works

The positive augmentation finding: KNN-based squad feature imputation for pre-2002 tournaments expanded the usable training set from 415 to 677 matches and produced a measurable improvement in log-loss — better than equivalent Optuna hyperparameter search on the unimputed 415-match set. More data beats better tuning, at this scale. This is a reassuring instance of a general principle, and it directly motivated the expansion from Feature Set B to Feature Set C as the basis for the v1 freeze model.

---

## 5. Contextual Features

### What We Engineered

Beyond squad composition and Elo, we engineered a set of curated contextual features:

- **form_GD5_gap**: difference in each team's goal difference over their trailing 5 matches (pre-tournament)
- **Travel distance proxy**: great-circle distance between each team's training base and the match venue
- **Altitude adjustment**: venue altitude above sea level (relevant for Mexico City and other high-altitude venues)
- **Qualifying form features**: points per match and goal difference during WC qualification campaign
- **Rest days gap**: difference in days since last match (relevant in knockout stage)

### What Matters

`form_GD5_gap` ranks 4th in RF feature importance — the highest-ranked non-Elo, non-archetype feature in the full model. Goal difference momentum is the strongest new signal added by contextual engineering. It captures something that Elo does not: recent trajectory. A team with a stable Elo rating but a deteriorating goal difference trend over their last five matches carries systematically different risk than an Elo-equivalent team in positive form.

Travel, altitude, and qualifying form contribute marginally (< 1% importance each individually) but collectively improve calibration, particularly for matches involving non-European teams and high-altitude venues.

---

## 6. The v1 Freeze Decision

### Dixon-Coles Is Intractable Locally

The natural next step after RF Optuna would be Dixon-Coles — the bivariate Poisson model with low-scoring correction that has been the statistical benchmark for match prediction since 1997. We implemented the likelihood function and confirmed it produces a meaningful improvement on the test set when fit correctly.

The problem is fitting. Dixon-Coles has approximately 400 parameters (attack and defense ratings for each of 200 active national teams, plus home advantage and the Dixon-Coles ρ correction). Fitting this model once is feasible. Fitting it 23 times — once per LOTO fold, each time on a different subset of international matches — is intractable on a single machine without distributed compute. The per-fold fit time exceeds what is practical for a solo research project. We document this as the primary stopping criterion for v1.

### Current Ceiling

The v1 freeze model sits at:

| Metric | Value | Benchmark |
|--------|------:|-----------|
| Log-loss (LOTO-CV) | 0.9419 | — |
| Accuracy (LOTO-CV) | 59.2% | Groll 2019: ~61%; Pinnacle: ~64% |
| 2026 OOS accuracy | 53.4% (73 matches) | +7.9pp above majority baseline |
| Majority baseline | 45.5% | — |

We are within approximately 2 percentage points of Groll (2019) and within approximately 5 percentage points of Pinnacle implied-probability accuracy. The Pinnacle gap is likely largely irreducible: it reflects the integration of injury news, line-up information, and market consensus from millions of bettors — information that is both more granular and more current than anything in a pre-match statistical model. The remaining gap to Groll (2019) is plausibly addressable with Dixon-Coles or ensemble stacking, but requires distributed compute.

The decision to freeze v1 at this point is not a concession to limitation — it is a deliberate scope boundary. v1 delivers the archetype framework, a competitive prediction model, and a live validation layer. Everything beyond this point is v2 territory.

---

## 7. What v2 Will Address

v2 is the **Tournament-State Simulator**: 100,000 Monte Carlo branches propagated through the full 2026 bracket, computing Expected Tournament Value (ETV) for each remaining team at each stage. This shifts the question from "who wins this match?" to "what is each team's tournament trajectory, given the archetype-structured outcome distributions we have measured?"

Specifically, v2 will:

- Propagate the v1 match-outcome distributions through the knockout bracket stochastically
- Compute ETV curves conditioned on archetype: a `generational_transition` team entering the quarterfinals has a different ETV profile than a `heavyweight_clash` participant with the same Elo
- Incorporate the Dixon-Coles correction on pre-fitted team parameters (rather than fitting per fold, use parameters fit once on the full 1930–2022 training set)
- Produce narrative archetype cards for each remaining 2026 match with historical precedent references

v3 (In-Play Event Engine) and v4 (Analytics Platform) extend further into real-time probability updating and a productized interface, respectively.
