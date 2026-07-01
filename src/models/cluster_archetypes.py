"""
Phase 7: Unsupervised Archetype Discovery
==========================================
Applies 5 unsupervised methods to WC matchup features to discover
latent cluster structure independent of the rule-based labels.

Methods:
  1. K-Means          (centroid-based, varying k=3..10)
  2. Gaussian Mixture (probabilistic, soft assignments)
  3. Hierarchical     (agglomerative, Ward linkage)
  4. HDBSCAN          (density-based, noise-aware)
  5. NMF              (non-negative matrix factorization, interpretable components)

Evaluation:
  - Silhouette score (internal)
  - Calinski-Harabasz (internal)
  - Adjusted Rand Index vs. rule-based labels (external)
  - Cluster stability across 100 bootstrap resamples (k-means only)

Output: data/processed/
  cluster_assignments.csv  -- cluster label per match per method
  cluster_profiles.csv     -- feature means per cluster per method
  cluster_eval.csv         -- evaluation metrics per method/k
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.constants import PROCESSED_DIR

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 adjusted_rand_score)
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[ERROR] scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("[WARN] hdbscan not installed — skipping HDBSCAN. Install with: pip install hdbscan")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed — skipping UMAP. Install with: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE SELECTION FOR CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

# Use features available across most WC matches (Elo = 100%, squad = 43%+)
ELO_FEATURES = [
    "elo_home_pre", "elo_away_pre", "elo_gap_abs", "win_prob_home",
]

SQUAD_FEATURES = [
    "age_mean_gap_abs", "h_age_mean", "a_age_mean",
    "top5_share_gap_abs", "h_top5_share", "a_top5_share",
    "league_div_gap",
]

CONTEXT_FEATURES = [
    "is_knockout", "is_neutral",
]

ARCHETYPE_NAMES = [
    "heavyweight_clash", "favorite_vs_underdog", "host_pressure",
    "generational_transition", "club_power_mismatch", "tactical_contrast",
    "knockout_volatility", "upset_realized",
]

K_RANGE = range(3, 11)
N_COMPONENTS_NMF = 6
BOOTSTRAP_RUNS = 100
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & PREPARE
# ─────────────────────────────────────────────────────────────────────────────

def load_wc_features():
    mf = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)

    wc = mf[mf["is_world_cup"] == True].copy()
    wc = wc.merge(arch[["match_id"] + ARCHETYPE_NAMES], on="match_id", how="left")

    print(f"  WC matches: {len(wc):,}")
    return wc


def prepare_feature_matrix(wc: pd.DataFrame, feature_set: str = "elo"):
    """
    Prepare scaled feature matrix for clustering.
    feature_set: 'elo' | 'squad' | 'all'
    """
    if feature_set == "elo":
        cols = ELO_FEATURES + CONTEXT_FEATURES
    elif feature_set == "squad":
        cols = SQUAD_FEATURES + CONTEXT_FEATURES
    else:  # all
        cols = ELO_FEATURES + SQUAD_FEATURES + CONTEXT_FEATURES

    avail = [c for c in cols if c in wc.columns]
    X_raw = wc[avail].copy()

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    print(f"  Feature matrix: {X_scaled.shape[0]} rows × {X_scaled.shape[1]} cols")
    print(f"  Features: {avail}")
    missing_pct = X_raw.isna().mean()
    if missing_pct.max() > 0.1:
        print(f"  [WARN] High missingness: {missing_pct[missing_pct>0.1].to_dict()}")

    return X_scaled, avail, imputer, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERING METHODS
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X, k_range, rule_labels_binary):
    """K-Means across k values. Returns best labels + eval table."""
    print(f"\n  [1] K-Means (k={k_range.start}..{k_range.stop-1})")
    records = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        ch  = calinski_harabasz_score(X, labels)

        # ARI vs. each rule-based archetype
        ari_scores = {}
        for col in ARCHETYPE_NAMES:
            if col in rule_labels_binary.columns:
                ari_scores[f"ari_{col}"] = adjusted_rand_score(
                    rule_labels_binary[col].fillna(0).astype(int), labels
                )

        records.append({
            "method": "kmeans", "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "inertia": round(km.inertia_, 2),
            **ari_scores,
        })
        all_labels[k] = labels
        print(f"    k={k}: sil={sil:.4f}, CH={ch:.1f}")

    # Bootstrap stability for best k (by silhouette)
    best_k = max(records, key=lambda r: r["silhouette"])["k"]
    print(f"  Best k by silhouette: {best_k}")

    return pd.DataFrame(records), all_labels, best_k


def run_gmm(X, k_range, rule_labels_binary):
    """Gaussian Mixture Models."""
    print(f"\n  [2] Gaussian Mixture (k={k_range.start}..{k_range.stop-1})")
    records = []
    all_labels = {}

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_SEED, n_init=5,
                              covariance_type="full")
        labels = gmm.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        bic = gmm.bic(X)

        records.append({
            "method": "gmm", "k": k,
            "silhouette": round(sil, 4),
            "bic": round(bic, 2),
            "calinski_harabasz": round(calinski_harabasz_score(X, labels), 2),
        })
        all_labels[k] = labels
        print(f"    k={k}: sil={sil:.4f}, BIC={bic:.1f}")

    best_k = min(records, key=lambda r: r["bic"])["k"]
    print(f"  Best k by BIC: {best_k}")
    return pd.DataFrame(records), all_labels, best_k


def run_hierarchical(X, k_range, rule_labels_binary):
    """Agglomerative clustering (Ward linkage)."""
    print(f"\n  [3] Hierarchical / Agglomerative (Ward linkage)")
    records = []
    all_labels = {}

    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        ch  = calinski_harabasz_score(X, labels)

        records.append({
            "method": "hierarchical", "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
        })
        all_labels[k] = labels
        print(f"    k={k}: sil={sil:.4f}, CH={ch:.1f}")

    best_k = max(records, key=lambda r: r["silhouette"])["k"]
    print(f"  Best k by silhouette: {best_k}")
    return pd.DataFrame(records), all_labels, best_k


def run_hdbscan(X):
    """HDBSCAN density-based clustering."""
    print(f"\n  [4] HDBSCAN (density-based)")
    if not HAS_HDBSCAN:
        print("  HDBSCAN not available — skipping")
        return pd.DataFrame(), {}

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5,
                                 metric="euclidean")
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    valid = labels != -1
    sil = silhouette_score(X[valid], labels[valid]) if valid.sum() > 1 and n_clusters > 1 else 0

    print(f"  Found {n_clusters} clusters, {n_noise} noise points, sil={sil:.4f}")
    record = pd.DataFrame([{
        "method": "hdbscan", "k": n_clusters,
        "silhouette": round(sil, 4),
        "n_noise": n_noise,
    }])
    return record, {"hdbscan": labels}


def run_nmf(X_raw_nonneg, n_components):
    """NMF for interpretable component analysis."""
    print(f"\n  [5] NMF ({n_components} components)")

    # NMF requires non-negative input — use MinMaxScaler
    scaler = MinMaxScaler()
    X_nn = scaler.fit_transform(X_raw_nonneg)

    nmf = NMF(n_components=n_components, random_state=RANDOM_SEED, max_iter=500)
    W = nmf.fit_transform(X_nn)  # match × component weights
    H = nmf.components_           # component × feature loadings

    # Assign cluster as dominant component
    labels = W.argmax(axis=1)
    n_unique = len(set(labels))
    sil = silhouette_score(X_nn, labels) if n_unique > 1 else 0

    print(f"  NMF sil={sil:.4f}, reconstruction error={nmf.reconstruction_err_:.4f}")
    record = pd.DataFrame([{
        "method": "nmf", "k": n_components,
        "silhouette": round(sil, 4),
        "reconstruction_err": round(nmf.reconstruction_err_, 4),
    }])
    return record, {"nmf": labels}, W, H


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTER PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_profiles(wc: pd.DataFrame, cluster_col: str,
                           method: str, k: int) -> pd.DataFrame:
    """Compute mean feature values per cluster."""
    feat_cols = [c for c in ELO_FEATURES + SQUAD_FEATURES + ARCHETYPE_NAMES
                 if c in wc.columns]
    profiles = wc.groupby(cluster_col)[feat_cols].mean().round(4).reset_index()
    profiles.insert(0, "method", method)
    profiles.insert(1, "k", k)
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
#  BOOTSTRAP STABILITY
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_stability(X, k, n_runs=BOOTSTRAP_RUNS):
    """
    Estimate cluster stability via ARI between full-data and bootstrap solutions.
    High ARI → stable clusters; low ARI → unstable.
    """
    full_labels = KMeans(n_clusters=k, random_state=RANDOM_SEED,
                         n_init=20).fit_predict(X)
    aris = []
    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_runs):
        idx = rng.choice(len(X), size=len(X), replace=True)
        boot_labels = KMeans(n_clusters=k, random_state=RANDOM_SEED,
                             n_init=10).fit_predict(X[idx])
        # Map boot labels back to full-data positions
        full_sub = full_labels[idx]
        aris.append(adjusted_rand_score(full_sub, boot_labels))

    mean_ari = float(np.mean(aris))
    print(f"  Bootstrap stability (k={k}, n={n_runs}): ARI={mean_ari:.4f} ± {np.std(aris):.4f}")
    return mean_ari, float(np.std(aris))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 7: Unsupervised Clustering")
    print("=" * 60)

    wc = load_wc_features()

    # Use Elo features (full coverage) as primary; squad features as secondary
    print("\n  Preparing Elo-based feature matrix (full WC coverage)...")
    X_elo, elo_cols, _, _ = prepare_feature_matrix(wc, "elo")

    print("\n  Preparing combined feature matrix (2010+ full coverage)...")
    X_all, all_cols, _, _ = prepare_feature_matrix(wc, "all")

    rule_labels = wc[[c for c in ARCHETYPE_NAMES if c in wc.columns]].copy()

    # ── 1. K-Means (Elo features — full coverage) ────────────────────────────
    km_eval, km_labels, km_best_k = run_kmeans(X_elo, K_RANGE, rule_labels)

    # Bootstrap stability for best k
    print(f"\n  Running bootstrap stability (k={km_best_k}, {BOOTSTRAP_RUNS} runs)...")
    km_ari_mean, km_ari_std = bootstrap_stability(X_elo, km_best_k)

    # ── 2. GMM ───────────────────────────────────────────────────────────────
    gmm_eval, gmm_labels, gmm_best_k = run_gmm(X_elo, K_RANGE, rule_labels)

    # ── 3. Hierarchical ──────────────────────────────────────────────────────
    hier_eval, hier_labels, hier_best_k = run_hierarchical(X_elo, K_RANGE, rule_labels)

    # ── 4. HDBSCAN ───────────────────────────────────────────────────────────
    hdb_eval, hdb_labels = run_hdbscan(X_elo)

    # ── 5. NMF (all features, nonneg) ────────────────────────────────────────
    wc_nonneg = wc[[c for c in ELO_FEATURES + SQUAD_FEATURES + CONTEXT_FEATURES
                    if c in wc.columns]].copy()
    imputer_nn = SimpleImputer(strategy="median")
    X_nn_raw = imputer_nn.fit_transform(wc_nonneg)
    nmf_eval, nmf_labels, W_nmf, H_nmf = run_nmf(X_nn_raw, N_COMPONENTS_NMF)

    # ── ARI: k-means best k vs. rule labels ─────────────────────────────────
    print(f"\n  ARI: K-Means (k={km_best_k}) vs. rule-based archetypes:")
    km_best = km_labels[km_best_k]
    for col in ARCHETYPE_NAMES:
        if col in wc.columns:
            ari = adjusted_rand_score(wc[col].fillna(0).astype(int), km_best)
            print(f"    vs. {col:<28}: {ari:.4f}")

    # ── Build cluster assignments table ─────────────────────────────────────
    assignments = wc[["match_id", "date", "home_team", "away_team",
                       "result", "is_world_cup"]].copy()
    assignments[f"kmeans_k{km_best_k}"] = km_best
    assignments[f"gmm_k{gmm_best_k}"] = gmm_labels[gmm_best_k]
    assignments[f"hier_k{hier_best_k}"] = hier_labels[hier_best_k]
    if hdb_labels:
        assignments["hdbscan"] = hdb_labels["hdbscan"]
    assignments["nmf_dominant"] = nmf_labels["nmf"]

    # ── Cluster profiles (k-means best k) ────────────────────────────────────
    col_name = f"kmeans_k{km_best_k}"
    wc[col_name] = km_best
    profiles = build_cluster_profiles(wc, col_name, "kmeans", km_best_k)

    # ── Save ─────────────────────────────────────────────────────────────────
    all_eval = pd.concat([km_eval, gmm_eval, hier_eval, hdb_eval, nmf_eval],
                         ignore_index=True)
    all_eval["bootstrap_ari_mean"] = None
    all_eval.loc[(all_eval["method"] == "kmeans") & (all_eval["k"] == km_best_k),
                 "bootstrap_ari_mean"] = km_ari_mean

    assignments.to_csv(PROCESSED_DIR / "cluster_assignments.csv", index=False)
    profiles.to_csv(PROCESSED_DIR / "cluster_profiles.csv", index=False)
    all_eval.to_csv(PROCESSED_DIR / "cluster_eval.csv", index=False)

    # NMF components
    nmf_components = pd.DataFrame(
        H_nmf,
        columns=[c for c in wc_nonneg.columns],
        index=[f"component_{i}" for i in range(N_COMPONENTS_NMF)]
    ).round(4)
    nmf_components.to_csv(PROCESSED_DIR / "nmf_components.csv")

    print(f"\n  Saved:")
    print(f"    cluster_assignments.csv  : {len(assignments):,} rows")
    print(f"    cluster_profiles.csv     : {len(profiles):,} rows")
    print(f"    cluster_eval.csv         : {len(all_eval):,} rows")
    print(f"    nmf_components.csv       : {N_COMPONENTS_NMF} components")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  {'Method':<15} {'Best k':>6} {'Silhouette':>11} {'Notes'}")
    print(f"  {'-'*55}")
    bests = [
        ("kmeans",      km_best_k,  km_eval[km_eval['k']==km_best_k]['silhouette'].values[0],
         f"bootstrap ARI={km_ari_mean:.3f}±{km_ari_std:.3f}"),
        ("gmm",         gmm_best_k, gmm_eval[gmm_eval['k']==gmm_best_k]['silhouette'].values[0],
         f"best k by BIC"),
        ("hierarchical",hier_best_k,hier_eval[hier_eval['k']==hier_best_k]['silhouette'].values[0],
         "Ward linkage"),
    ]
    if not hdb_eval.empty:
        bests.append(("hdbscan", int(hdb_eval['k'].values[0]),
                      float(hdb_eval['silhouette'].values[0]),
                      f"{int(hdb_eval.get('n_noise', pd.Series([0])).values[0])} noise pts"))
    bests.append(("nmf",  N_COMPONENTS_NMF, float(nmf_eval['silhouette'].values[0]),
                  f"reconstruction err={nmf_eval['reconstruction_err'].values[0]:.4f}"))

    for method, k, sil, note in bests:
        print(f"  {method:<15} {k:>6} {sil:>11.4f}  {note}")

    print("\n" + "=" * 60)
    print("  Phase 7 complete ✓")
    print("=" * 60)

    return assignments, all_eval


if __name__ == "__main__":
    main()
