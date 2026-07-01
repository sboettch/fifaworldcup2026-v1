# NOTE — Research / Experimental Script
# ─────────────────────────────────────────────────────────────────────────────
# This script was part of the v1 investigation and documents findings but is
# NOT part of the core reproducible pipeline (`make pipeline`).
# See docs/checkin_2.md for results and interpretation.
# ─────────────────────────────────────────────────────────────────────────────

"""
Phase 8 Extended: Full 6-Family Model Ablation
================================================
Adds HyperNEAT, GA Dynamic Ensemble, and GoL Cellular Automaton to the
Phase 8 ablation table alongside the 4 classical families.

Runs LOTO-CV (leave-one-tournament-out) for all 6 families × 3 feature sets.
Outputs: model_results_extended.csv, feature_importance_extended.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.constants import PROCESSED_DIR
from src.models.hyperneat_classifier import HyperNEATClassifier
from src.models.ga_ensemble import GADynamicEnsemble, ARCHETYPE_COLS
from src.models.gol_classifier import GoLClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from sklearn.base import clone

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ELO_FEATURES = [
    "elo_home_pre", "elo_away_pre", "elo_gap", "elo_gap_abs",
    "win_prob_home", "win_prob_away", "k_factor",
    "is_knockout", "is_neutral",
]
ARCHETYPE_LABELS = [
    "heavyweight_clash", "favorite_vs_underdog", "host_pressure",
    "generational_transition", "club_power_mismatch", "tactical_contrast",
    "knockout_volatility",
]
SQUAD_FEATURES = [
    "h_age_mean", "a_age_mean", "age_mean_gap_abs",
    "h_top5_share", "a_top5_share", "top5_share_gap_abs",
    "league_div_gap",
]


def load_wc_data():
    mf   = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)
    arch_cols = ["match_id"] + [c for c in ARCHETYPE_LABELS if c in arch.columns]
    wc = mf[mf["is_world_cup"] == True].copy()
    wc = wc.merge(arch[arch_cols], on="match_id", how="left")
    wc["match_year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year
    wc["y_3class"] = wc["result"].map({"H": 0, "D": 1, "A": 2})
    return wc


def prep(X_train_raw, X_test_raw):
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    Xte = scaler.transform(imputer.transform(X_test_raw))
    return Xtr, Xte


def eval_metrics(y_true, preds, probs):
    acc  = accuracy_score(y_true, preds)
    ll   = log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7))
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), probs[:, c])
        for c in range(3)
    ])
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return acc, ll, brier, auc


def loto_cv_model(wc, feat_cols, model_name, model, arch_aware=False):
    """LOTO-CV for one model. arch_aware=True passes archetype context to GA ensemble."""
    tournaments = sorted(wc["match_year"].dropna().unique())
    tournaments = [t for t in tournaments if (wc["match_year"] == t).sum() >= 6]

    all_true, all_preds, all_probs = [], [], []

    for holdout in tournaments:
        train = wc[wc["match_year"] != holdout].copy()
        test  = wc[wc["match_year"] == holdout].copy()
        y_train = train["y_3class"].dropna()
        y_test  = test["y_3class"].dropna()
        Xtr_raw = train[feat_cols].loc[y_train.index]
        Xte_raw = test[feat_cols].loc[y_test.index]

        if len(y_train) < 20 or len(y_test) < 4:
            continue

        Xtr, Xte = prep(Xtr_raw, Xte_raw)

        try:
            if arch_aware and isinstance(model, GADynamicEnsemble):
                arch_train = train[[c for c in ARCHETYPE_COLS if c in train.columns]].loc[y_train.index].reset_index(drop=True)
                arch_test  = test[[c for c in ARCHETYPE_COLS if c in test.columns]].loc[y_test.index].reset_index(drop=True)
                m = clone(model)
                m.fit(Xtr, y_train.values, X_arch=arch_train)
                probs = m.predict_proba(Xte, X_arch=arch_test)
                preds = probs.argmax(axis=1)
            else:
                from sklearn.base import clone as sk_clone
                m = sk_clone(model)
                m.fit(Xtr, y_train.values)
                probs = m.predict_proba(Xte)
                preds = m.predict(Xte)
        except Exception as e:
            print(f"    [WARN] {holdout}/{model_name} failed: {str(e)[:60]}")
            continue

        all_true.extend(y_test.values)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    if not all_true:
        return {}

    yt = np.array(all_true)
    yp = np.array(all_preds)
    pr = np.array(all_probs)
    acc, ll, brier, auc = eval_metrics(yt, yp, pr)
    return {
        "model": model_name,
        "n_matches": len(yt),
        "n_tournaments": len(tournaments),
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "auc_macro": round(auc, 4) if not np.isnan(auc) else None,
    }


def main():
    print("=" * 60)
    print("  Phase 8 Extended: All 6 Model Families")
    print("=" * 60)

    wc = load_wc_data()

    feat_A = [c for c in ELO_FEATURES if c in wc.columns]
    feat_B = [c for c in ELO_FEATURES + ARCHETYPE_LABELS if c in wc.columns]
    feat_C = [c for c in ELO_FEATURES + ARCHETYPE_LABELS + SQUAD_FEATURES if c in wc.columns]

    print(f"  WC matches: {len(wc):,}")
    print(f"  Feature A (Elo): {len(feat_A)}, B (Elo+Arch): {len(feat_B)}, C (Full): {len(feat_C)}")

    # Classical models (from Phase 8)
    classical = {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
            random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.15, alpha=0.01),
    }

    # Novel models
    novel = {
        "hyperneat": HyperNEATClassifier(
            n_hidden=12, n_generations=30, pop_size=40, random_state=RANDOM_SEED),
        "ga_dynamic_ensemble": GADynamicEnsemble(
            n_generations=50, pop_size=60, random_state=RANDOM_SEED),
        "gol_cellular_automaton": GoLClassifier(
            n_generations=40, pop_size=50, random_state=RANDOM_SEED),
    }

    results = []

    for feat_name, feat_cols in [("A_elo_only", feat_A),
                                   ("B_elo_archetypes", feat_B),
                                   ("C_full", feat_C)]:
        print(f"\n  Feature set {feat_name}:")

        print("    [Classical models]")
        for name, model in classical.items():
            print(f"      {name}...", end="", flush=True)
            r = loto_cv_model(wc, feat_cols, name, model)
            if r:
                r["feature_set"] = feat_name
                r["model_family"] = "classical"
                results.append(r)
                print(f" ll={r['log_loss']:.4f} acc={r['accuracy']:.3f}")

        print("    [Novel models]")
        for name, model in novel.items():
            print(f"      {name}...", end="", flush=True)
            arch_aware = (name == "ga_dynamic_ensemble")
            r = loto_cv_model(wc, feat_cols, name, model, arch_aware=arch_aware)
            if r:
                r["feature_set"] = feat_name
                r["model_family"] = "novel"
                results.append(r)
                print(f" ll={r['log_loss']:.4f} acc={r['accuracy']:.3f}")

    results_df = pd.DataFrame(results)

    # ── Full ablation table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FULL ABLATION TABLE (all 6 families × 3 feature sets)")
    print("=" * 70)
    print(f"\n  {'Model':<28} {'Family':<10} {'Features':<22} {'LogLoss':>9} {'Acc':>7} {'AUC':>7}")
    print(f"  {'-'*82}")

    for fs in ["A_elo_only", "B_elo_archetypes", "C_full"]:
        sub = results_df[results_df["feature_set"] == fs].sort_values("log_loss")
        print(f"\n  --- {fs} ---")
        for _, row in sub.iterrows():
            auc = f"{row['auc_macro']:.4f}" if row.get("auc_macro") else "   n/a"
            fam = "★" if row["model_family"] == "novel" else " "
            print(f"  {fam}{str(row['model']):<28} {str(row['model_family']):<10} "
                  f"{fs:<22} {row['log_loss']:>9.4f} {row['accuracy']:>7.4f} {auc:>7}")

    # ── Archetype lift: best novel vs best classical ──────────────────────────
    print("\n  Novel vs. Classical — archetype lift (Feature set B):")
    b_classic = results_df[(results_df["feature_set"] == "B_elo_archetypes") &
                           (results_df["model_family"] == "classical")]
    b_novel   = results_df[(results_df["feature_set"] == "B_elo_archetypes") &
                           (results_df["model_family"] == "novel")]

    best_classic_ll = b_classic["log_loss"].min() if len(b_classic) else float("inf")
    best_novel_ll   = b_novel["log_loss"].min() if len(b_novel) else float("inf")
    best_classic    = b_classic.sort_values("log_loss").iloc[0]["model"] if len(b_classic) else "n/a"
    best_novel_name = b_novel.sort_values("log_loss").iloc[0]["model"] if len(b_novel) else "n/a"

    print(f"    Best classical: {best_classic:<28} log-loss={best_classic_ll:.4f}")
    print(f"    Best novel    : {best_novel_name:<28} log-loss={best_novel_ll:.4f}")
    lift = best_classic_ll - best_novel_ll
    direction = "✓ Novel improves" if lift > 0 else "✗ Classical better"
    print(f"    Δ log-loss = {lift:+.4f}  {direction}")

    # ── Save ─────────────────────────────────────────────────────────────────
    results_df.to_csv(PROCESSED_DIR / "model_results_extended.csv", index=False)

    # Update model_summary.json with best-of-all result
    best_overall = results_df.sort_values("log_loss").iloc[0]
    summary_ext = {
        "best_model_name": best_overall["model"],
        "best_feature_set": best_overall["feature_set"],
        "best_log_loss": float(best_overall["log_loss"]),
        "best_accuracy": float(best_overall["accuracy"]),
        "n_model_families": 6,
        "n_wc_matches": int(results_df["n_matches"].max()),
        "novel_vs_classical_lift_B": round(lift, 4),
        "novel_best_model": best_novel_name,
        "novel_best_ll": float(best_novel_ll) if best_novel_ll != float("inf") else None,
    }
    with open(PROCESSED_DIR / "model_summary_extended.json", "w") as f:
        json.dump(summary_ext, f, indent=2)

    print(f"\n  Saved: model_results_extended.csv ({len(results_df)} rows)")
    print(f"         model_summary_extended.json")

    print("\n" + "=" * 60)
    print("  Phase 8 Extended complete ✓")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
