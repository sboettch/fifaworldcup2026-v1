# Check-in 1: Data Pipeline through Archetype Framework
**FIFA World Cup 2026 — Matchup Archetype Classification**
*sophia · Phases 1–5*

---

## 1. What We Set Out to Do

The central question of this project is deceptively simple: do World Cup matchups have *types*, and if so, do those types help predict outcomes? The hypothesis is that a heavyweight clash between two top-Elo nations behaves differently — in outcome distribution, upset rate, and volatility — from a structural favorite-vs-underdog match with the same Elo gap. If so, an intermediate representation of the matchup's *character* should carry predictive information that raw feature vectors obscure.

To test this, we needed to do three things in sequence: build a clean longitudinal dataset covering nearly a century of World Cup football, engineer matchup features that capture structural character rather than just scalar ratings, and define a principled archetype taxonomy. Check-in 1 covers all of that ground.

---

## 2. Data Collection and Harmonization

### The Fjelstul Dataset and Its Limits

The backbone of the dataset is the Fjelstul World Cup Database, which covers all 23 WC editions from 1930 to 2022 — 1,037 matches across 86 nations. For match results, scores, stage information, and player rosters, it is high-quality and nearly complete. Entity resolution is the messier problem: teams change names (Zaire → DR Congo), split (Yugoslavia → Serbia, Croatia, Bosnia, etc.), and appear under variant spellings across sources. We maintain a `map_team_names` table with 424 cross-source mappings, and flag historical teams with `is_historical` and `modern_successor` fields.

Quality issues in the Fjelstul data cluster around two areas. Squad demographics — birth dates, caps, club affiliations — are sparse before 1950 and unreliable before 1970. Pre-1950 birth dates are frequently missing or approximate, which means derived features like squad age and international experience cannot be trusted for the early tournament era. We handle this by flagging per-feature coverage boundaries and reporting results stratified by era.

The full match history — 49,493 international matches from 1872 to 2026 — comes from the Jürisoo dataset. This is the foundation for the Elo reconstruction: we cannot compute meaningful pre-match Elo ratings using only the 1,037 WC matches, so we rebuild Elo across all international football.

### The Effective Training Set Problem

The single most consequential data finding of Phase 1 is the gap between the nominal dataset size and the effective training set. On paper, we have 1,037 WC matches. In practice, squad-level features — player demographics, club affiliations, league diversity — are only available from 2002 onwards via Transfermarkt enrichment. The effective full-feature training set (Elo + archetypes + squad features) is 415–677 matches, not 1,037.

That gap matters enormously for model selection. A dataset of 415–677 matches in 23 temporal folds is a fundamentally different statistical regime than 1,037 matches would be. Neural networks and evolutionary methods were designed with the larger number in mind; the actual constraint shapes almost every downstream decision.

### 2026 Live Data

The 2026 layer is scraped from Wikipedia squad pages (automated via GitHub Actions, updating every 6 hours), supplemented by openfootball and FIFA result feeds. As of July 1, 2026, 73 matches are scored and 1,290 player entries across all 48 participating nations are loaded. The automated collector continues through the final on July 19.

---

## 3. Feature Engineering: 51 Pairwise Features

Each match is represented by a team fingerprint vector for each side, then collapsed into pairwise matchup features capturing *differences* rather than individual team characteristics. The final feature matrix contains 51 pairwise features across three conceptual groups:

**Elo-based strength features**: Pre-tournament Elo ratings, Elo gap, whether both teams are in the top Elo quartile. These have 100% coverage across the full match history and form the backbone of Feature Set A.

**Squad composition features**: Mean squad age, international caps distribution, top-5 league representation (EPL, La Liga, Bundesliga, Serie A, Ligue 1), league diversity entropy, returning-player ratio (players who appeared in a prior WC). These require the Transfermarkt enrichment layer and are therefore only reliable from 2002 onwards.

**Contextual features**: Host-nation flag, home-continent advantage, knockout-stage indicator, travel distance proxy, and (for modern tournaments) form-based momentum features including `form_GD5_gap` — the difference in goal-difference momentum over the trailing five matches. Feature selection uses VIF screening (threshold: VIF > 5) and domain-constrained ablation to avoid inflating the feature count with correlated proxies.

The feature pipeline achieves 100% coverage for the Elo subset (Feature Set A), dropping to the 2002+ era for the full squad + contextual sets (Feature Sets B and C).

---

## 4. The Archetype Framework Decision: Rule-Based vs. Clustering

### What Clustering Told Us

Before committing to rule-based archetypes, we ran all five unsupervised clustering methods on the 1,037-match dataset to see whether the data would organize itself into interpretable groups. The results were instructive but not decisive.

The best silhouette score — 0.360, achieved by HDBSCAN with k = 4 — is reasonable but not compelling. K-Means at k = 5 reaches silhouette = 0.3365 with excellent bootstrap stability (ARI = 0.916 ± 0.042 across subsamples), and hierarchical clustering (Ward linkage) agrees on k = 4. Three independent methods converging on k = 4–5 is a real finding: there is latent structure in the WC matchup space at roughly that granularity.

The problem is alignment. The maximum ARI between any unsupervised cluster and any rule-based archetype label is 0.38, corresponding to `favorite_vs_underdog`. All other rule-based labels have ARI below 0.17. The data-driven clusters organize primarily around a *prestige/era* dimension — separating high-Elo from low-Elo matchups across historical and modern eras — rather than around the structural interaction types we care about. A cluster labeled "high Elo, modern era" is not the same as "heavyweight clash in a knockout context," even if the overlap is substantial.

GMM at k = 10 (BIC-preferred) produces finer granularity but silhouette drops to 0.1733; NMF at k = 6 yields silhouette = 0.0367 and is useful only for component interpretability, not as a standalone taxonomy.

### Why Rule-Based Won

Three considerations drove the decision toward rule-based archetypes:

1. **Coverage**: The rule-based taxonomy achieves 99.6% coverage for 2014–2026 matches (dropping to 82% for pre-1970, where squad features are sparse). HDBSCAN classifies only 1 match as noise, but the labels it produces are not interpretable in domain terms.

2. **Interpretability**: "This match is classified as `heavyweight_clash` because both teams are in the top Elo quartile and reached the semifinal in their prior WC appearance" is actionable. A cluster ID derived from UMAP-projected PCA space is not.

3. **Domain alignment**: The research question is about *types of matchups that matter for outcomes*, not about discovering whatever structure the data contains. Rule-based labels directly encode domain knowledge about which structural features drive different outcome distributions.

The clustering results are retained as a validation check rather than discarded — the k = 4–5 convergence is itself a finding about WC matchup space that motivates future work.

---

## 5. Unsupervised Results Summary

Five methods were applied to the 1,037 WC match dataset using six Elo-based features (full coverage):

| Method | Best k | Silhouette | Key finding |
|--------|:------:|:----------:|-------------|
| K-Means | 5 | 0.3365 | Bootstrap stability ARI = 0.916 ± 0.042 |
| Hierarchical (Ward) | 4 | 0.3173 | Agrees with HDBSCAN on k = 4 |
| HDBSCAN | 4 | **0.3595** | Only 1 noise point; near-universal coverage |
| GMM | 10 | 0.1733 | BIC prefers finer granularity |
| NMF | 6 | 0.0367 | Used for component interpretability only |

**Cross-method consensus**: k = 4–5 is the stable range. The HDBSCAN result (sil = 0.360, k = 4, one noise point) is the recommended clustering solution. Max ARI between any cluster and rule-based label: 0.38 (`favorite_vs_underdog`); all others < 0.17.

UMAP visualizations confirm that rule-based archetypes do not form tight, separable regions in the low-dimensional projection — they overlap in ways consistent with their non-exclusive design. The overlap between `heavyweight_clash` and `knockout_volatility` is particularly pronounced, as expected from their co-occurrence logic.

---

## 6. Key Open Question Going into Check-in 2

The archetype framework exists. The question it was designed to answer does not yet have an answer: **do archetype labels actually improve outcome prediction above a strong Elo baseline?**

The unsupervised results show that WC matchup space has real structure (k = 4–5 is stable, silhouette is positive), but that the rule-based taxonomy captures it only partially (max ARI = 0.38). It is entirely plausible that archetype labels are too correlated with the underlying Elo features to carry incremental signal — that a model with the raw Elo inputs already captures whatever the archetype label encodes.

Answering this requires a formal ablation across model families and feature sets, evaluated under leave-one-tournament-out cross-validation across all 23 WC editions. That is the work of Check-in 2.
