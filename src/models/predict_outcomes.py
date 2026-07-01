"""
Phase 8: Supervised Prediction Models
=======================================
The core ablation experiment: does adding archetype labels improve
outcome prediction beyond Elo alone?

Model A: Elo features only  (baseline)
Model B: Elo + archetype labels  (hypothesis)

Model families:
  1. Logistic Regression (L2)
  2. Random Forest
  3. XGBoost / Gradient Boosting
  4. MLP (neural network)

Evaluation:
  - Leave-one-tournament-out cross-validation (LOTO-CV)
  - Metrics: log-loss, Brier score, accuracy, ROC-AUC
  - Calibration: reliability diagrams
  - Ablation table: Model A vs. Model B per model family

Outcome variable: result (H/D/A) as 3-class or binary home-win
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score,
                              roc_auc_score, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
import json

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

ELO_FEATURES = [
    "elo_home_pre", "elo_away_pre", "elo_gap", "elo_gap_abs",
    "win_prob_home", "win_prob_away", "k_factor",
    "is_knockout", "is_neutral",
]

ARCHETYPE_LABELS = [
    "heavyweight_clash", "favorite_vs_underdog", "host_pressure",
    "generational_transition", "club_power_mismatch", "tactical_contrast",
    "knockout_volatility",
    # Note: upset_realized is post-hoc — excluded as predictor
]

SQUAD_FEATURES = [
    "h_age_mean", "a_age_mean", "age_mean_gap_abs",
    "h_top5_share", "a_top5_share", "top5_share_gap_abs",
    "league_div_gap",
]

RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_wc_data():
    mf = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)

    wc = mf[mf["is_world_cup"] == True].copy()
    arch_cols = ["match_id"] + ARCHETYPE_LABELS
    arch_cols = [c for c in arch_cols if c in arch.columns]
    wc = wc.merge(arch[arch_cols], on="match_id", how="left")

    # Extract tournament year for LOTO-CV
    wc["match_year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year

    # Binary outcome: home win (1) or not (0)
    wc["y_binary"] = (wc["result"] == "H").astype(int)

    # 3-class outcome
    result_map = {"H": 0, "D": 1, "A": 2}
    wc["y_3class"] = wc["result"].map(result_map)

    print(f"  WC matches: {len(wc):,}")
    print(f"  Outcome distribution: H={( wc['result']=='H').mean():.1%}, "
          f"D={(wc['result']=='D').mean():.1%}, A={(wc['result']=='A').mean():.1%}")

    return wc


# ─────────────────────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=500, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15,
            alpha=0.01
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LEAVE-ONE-TOURNAMENT-OUT CV
# ─────────────────────────────────────────────────────────────────────────────

def loto_cv(wc: pd.DataFrame, feature_cols: list, model_name: str, model,
            outcome: str = "y_3class") -> dict:
    """
    Leave-one-tournament-out cross-validation.
    Each WC edition is held out once; model trained on all others.
    """
    tournaments = sorted(wc["match_year"].dropna().unique())
    # Only use editions where we have enough data
    tournaments = [t for t in tournaments if
                   (wc["match_year"] == t).sum() >= 6]

    all_preds = []
    all_true  = []
    all_probs = []
    fold_records = []

    for holdout_year in tournaments:
        train = wc[wc["match_year"] != holdout_year].copy()
        test  = wc[wc["match_year"] == holdout_year].copy()

        if len(train) < 50 or len(test) < 4:
            continue

        # Prepare features
        X_train_raw = train[feature_cols].copy()
        X_test_raw  = test[feature_cols].copy()
        y_train = train[outcome].dropna()
        y_test  = test[outcome].dropna()

        # Align indices
        X_train_raw = X_train_raw.loc[y_train.index]
        X_test_raw  = X_test_raw.loc[y_test.index]

        if len(y_train) < 20 or len(y_test) < 4:
            continue

        # Impute + scale
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_raw)
        X_test  = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Fit
        m = clone_model(model)
        try:
            m.fit(X_train, y_train.values)
            probs = m.predict_proba(X_test)
            preds = m.predict(X_test)
        except Exception as e:
            print(f"    [WARN] {holdout_year} failed: {e}")
            continue

        fold_acc = accuracy_score(y_test.values, preds)
        fold_ll = log_loss(y_test.values, probs, labels=[0, 1, 2])
        fold_brier = np.mean([
            brier_score_loss((y_test.values == c).astype(int), probs[:, c])
            for c in range(probs.shape[1])
        ])
        try:
            fold_auc = roc_auc_score(
                y_test.values, probs, multi_class="ovr", average="macro"
            )
        except Exception:
            fold_auc = np.nan

        fold_records.append({
            "holdout_year": int(holdout_year),
            "n_matches": int(len(y_test)),
            "accuracy": round(float(fold_acc), 4),
            "log_loss": round(float(fold_ll), 4),
            "brier_score": round(float(fold_brier), 4),
            "auc_macro": round(float(fold_auc), 4) if not np.isnan(fold_auc) else None,
        })

        all_true.extend(y_test.values.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    if not all_true:
        return {}

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_true, all_preds)
    ll  = log_loss(all_true, all_probs)

    # Brier score (multiclass: mean over classes)
    n_classes = all_probs.shape[1]
    brier = np.mean([
        brier_score_loss((all_true == c).astype(int), all_probs[:, c])
        for c in range(n_classes)
    ])

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_true, all_probs, multi_class="ovr",
                            average="macro")
    except Exception:
        auc = np.nan

    return {
        "model": model_name,
        "n_matches": len(all_true),
        "n_tournaments": len(tournaments),
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "auc_macro": round(auc, 4) if not np.isnan(auc) else None,
        "folds": fold_records,
    }


def clone_model(model):
    """Deep copy a sklearn model."""
    from sklearn.base import clone
    return clone(model)


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(wc: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Train a gradient boosting model on all data and extract feature importances."""
    X_raw = wc[feature_cols].copy()
    y = wc["y_3class"].dropna()
    X_raw = X_raw.loc[y.index]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED
    )
    gb.fit(X, y.values)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False).round(4)

    return imp


# ─────────────────────────────────────────────────────────────────────────────
#  MAJORITY CLASS BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def majority_baseline(wc: pd.DataFrame) -> dict:
    """Predict the most common class for every match."""
    y = wc["y_3class"].dropna()
    majority = int(y.mode()[0])
    preds = np.full(len(y), majority)
    # Probability = uniform (worst case) or one-hot to majority
    # Use soft version: class frequency
    freqs = y.value_counts(normalize=True).sort_index()
    probs = np.tile(freqs.values, (len(y), 1))

    acc = accuracy_score(y, preds)
    ll  = log_loss(y, probs)
    n_classes = 3
    brier = np.mean([
        brier_score_loss((y == c).astype(int), probs[:, c])
        for c in range(n_classes)
    ])
    return {
        "model": "majority_class_baseline",
        "n_matches": len(y),
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "auc_macro": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 8: Supervised Prediction Models")
    print("=" * 60)

    wc = load_wc_data()
    models = get_models()

    # ── Feature sets ──────────────────────────────────────────────────────────
    feat_A = [c for c in ELO_FEATURES if c in wc.columns]
    feat_B = [c for c in ELO_FEATURES + ARCHETYPE_LABELS if c in wc.columns]
    feat_C = [c for c in ELO_FEATURES + ARCHETYPE_LABELS + SQUAD_FEATURES
              if c in wc.columns]

    print(f"\n  Feature set A (Elo only)      : {len(feat_A)} features")
    print(f"  Feature set B (Elo+Archetypes): {len(feat_B)} features")
    print(f"  Feature set C (Elo+Arch+Squad): {len(feat_C)} features")
    print(f"\n  Running LOTO-CV across {wc['match_year'].nunique()} WC editions...")

    results = []
    fold_results = []

    # Majority class baseline
    results.append({**majority_baseline(wc), "feature_set": "baseline"})

    # Elo-only logistic regression baseline
    print("\n  [A] Elo-only models:")
    for name, model in models.items():
        print(f"    {name}...", end="", flush=True)
        r = loto_cv(wc, feat_A, name, model)
        if r:
            folds = r.pop("folds", [])
            r["feature_set"] = "A_elo_only"
            results.append(r)
            for fold in folds:
                fold_results.append({**fold, "model": name, "feature_set": "A_elo_only"})
            print(f" log-loss={r['log_loss']:.4f}, acc={r['accuracy']:.3f}")

    print("\n  [B] Elo + Archetype labels:")
    for name, model in models.items():
        print(f"    {name}...", end="", flush=True)
        r = loto_cv(wc, feat_B, name, model)
        if r:
            folds = r.pop("folds", [])
            r["feature_set"] = "B_elo_archetypes"
            results.append(r)
            for fold in folds:
                fold_results.append({**fold, "model": name, "feature_set": "B_elo_archetypes"})
            print(f" log-loss={r['log_loss']:.4f}, acc={r['accuracy']:.3f}")

    print("\n  [C] Elo + Archetypes + Squad features:")
    for name, model in models.items():
        print(f"    {name}...", end="", flush=True)
        r = loto_cv(wc, feat_C, name, model)
        if r:
            folds = r.pop("folds", [])
            r["feature_set"] = "C_full"
            results.append(r)
            for fold in folds:
                fold_results.append({**fold, "model": name, "feature_set": "C_full"})
            print(f" log-loss={r['log_loss']:.4f}, acc={r['accuracy']:.3f}")

    results_df = pd.DataFrame(results)
    folds_df = pd.DataFrame(fold_results)

    # ── Ablation summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION RESULTS (3-class: H / D / A)")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Features':<22} {'LogLoss':>9} {'Brier':>7} {'Acc':>7} {'AUC':>7}")
    print(f"  {'-'*75}")
    for _, row in results_df.sort_values(["feature_set", "log_loss"]).iterrows():
        auc_str = f"{row['auc_macro']:.4f}" if row.get("auc_macro") else "  n/a"
        print(f"  {str(row['model']):<25} {str(row['feature_set']):<22} "
              f"{row['log_loss']:>9.4f} {row['brier_score']:>7.4f} "
              f"{row['accuracy']:>7.4f} {auc_str:>7}")

    # ── Archetype lift ────────────────────────────────────────────────────────
    print("\n  Archetype lift (Feature set B vs. A, by model):")
    for name in models.keys():
        a_row = results_df[(results_df["model"] == name) &
                           (results_df["feature_set"] == "A_elo_only")]
        b_row = results_df[(results_df["model"] == name) &
                           (results_df["feature_set"] == "B_elo_archetypes")]
        if len(a_row) and len(b_row):
            ll_lift = a_row["log_loss"].values[0] - b_row["log_loss"].values[0]
            direction = "✓ IMPROVED" if ll_lift > 0 else "✗ no gain"
            print(f"    {name:<25}: Δlog-loss={ll_lift:+.4f}  {direction}")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\n  Feature importance (gradient boosting on full dataset, feat set B):")
    imp_df = feature_importance(wc, feat_B)
    for _, row in imp_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df.to_csv(PROCESSED_DIR / "model_results.csv", index=False)
    imp_df.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    if not folds_df.empty:
        folds_df.to_csv(PROCESSED_DIR / "model_cv_folds.csv", index=False)
        metric_cols = ["accuracy", "log_loss", "brier_score", "auc_macro"]
        summary_rows = []
        for (feature_set, model), group in folds_df.groupby(["feature_set", "model"]):
            row = {
                "feature_set": feature_set,
                "model": model,
                "n_folds": int(group["holdout_year"].nunique()),
                "n_matches": int(group["n_matches"].sum()),
            }
            for metric in metric_cols:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(values):
                    row[f"{metric}_mean"] = round(float(values.mean()), 4)
                    row[f"{metric}_sd"] = round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0
            summary_rows.append(row)
        cv_summary_df = pd.DataFrame(summary_rows).sort_values(["feature_set", "log_loss_mean"])
        cv_summary_df.to_csv(PROCESSED_DIR / "model_cv_summary.csv", index=False)

    # Save summary JSON for proposal write-up
    best_model = results_df[results_df["feature_set"] == "B_elo_archetypes"].sort_values("log_loss").iloc[0]
    baseline   = results_df[results_df["feature_set"] == "A_elo_only"].sort_values("log_loss").iloc[0]
    summary = {
        "best_model_name": best_model["model"],
        "best_log_loss_B": float(best_model["log_loss"]),
        "best_log_loss_A": float(baseline["log_loss"]),
        "archetype_lift_log_loss": float(baseline["log_loss"] - best_model["log_loss"]),
        "best_accuracy_B": float(best_model["accuracy"]),
        "n_wc_matches": int(results_df["n_matches"].max()),
    }
    with open(PROCESSED_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved:")
    print(f"    model_results.csv     : {len(results_df)} rows")
    if not folds_df.empty:
        print(f"    model_cv_folds.csv    : {len(folds_df)} fold rows")
        print(f"    model_cv_summary.csv  : {len(cv_summary_df)} model summaries")
    print(f"    feature_importance.csv: {len(imp_df)} features")
    print(f"    model_summary.json")

    print("\n" + "=" * 60)
    print("  Phase 8 complete ✓")
    print("=" * 60)

    return results_df, imp_df


if __name__ == "__main__":
    main()
