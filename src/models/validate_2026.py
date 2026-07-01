"""
Phase 9: 2026 Out-of-Sample Validation
========================================
Generates timestamped prediction snapshots for 2026 World Cup matches
using the best model from Phase 8, then validates against actual results.

Important: a snapshot is only a genuine pre-match prediction when it is
generated and committed before the corresponding match kicks off. Snapshots
created after completed matches are retrospective validation artifacts.

Outputs:
  data/processed/predictions_2026.csv   — pre-match predictions (timestamped)
  data/processed/validation_2026.csv    — predictions vs. actual outcomes
  data/processed/validation_summary.json — key metrics for write-up
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.constants import PROCESSED_DIR

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42

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

FEATURE_COLS = [c for c in ELO_FEATURES + ARCHETYPE_LABELS]


def load_all_data():
    mf   = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)
    arch_cols = ["match_id"] + [c for c in ARCHETYPE_LABELS if c in arch.columns]
    mf = mf.merge(arch[arch_cols], on="match_id", how="left")
    mf["match_year"] = pd.to_datetime(mf["date"], errors="coerce").dt.year
    mf["y_3class"] = mf["result"].map({"H": 0, "D": 1, "A": 2})
    return mf


def train_best_model(historical: pd.DataFrame):
    """Train the selected Random Forest model on all pre-2026 WC data."""
    feat_cols = [c for c in FEATURE_COLS if c in historical.columns]
    y = historical["y_3class"].dropna()
    X_raw = historical[feat_cols].loc[y.index]

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw))

    params_path = PROCESSED_DIR / "best_model_params.json"
    if params_path.exists():
        with params_path.open(encoding="utf-8") as handle:
            params = json.load(handle)
    else:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": None,
        }

    model = RandomForestClassifier(
        **params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y.values)
    print(f"  Trained on {len(y)} historical WC matches")
    return model, imputer, scaler, feat_cols


def generate_predictions(model, imputer, scaler, feat_cols,
                          matches_2026: pd.DataFrame) -> pd.DataFrame:
    """Generate timestamped predictions for 2026 matches."""
    X_raw = matches_2026[feat_cols].copy()
    X = scaler.transform(imputer.transform(X_raw))
    probs = model.predict_proba(X)
    preds = model.predict(X)

    result_map = {0: "H", 1: "D", 2: "A"}
    label_map  = {0: "Home win", 1: "Draw", 2: "Away win"}

    out = matches_2026[["match_id", "date", "home_team", "away_team",
                          "stage", "is_knockout"]].copy()

    # Archetype context
    for col in ARCHETYPE_LABELS:
        if col in matches_2026.columns:
            out[col] = matches_2026[col].values

    snapshot_time = datetime.now(timezone.utc)
    out["pred_result"]      = [result_map[p] for p in preds]
    out["pred_label"]       = [label_map[p] for p in preds]
    out["prob_home_win"]    = probs[:, 0].round(4)
    out["prob_draw"]        = probs[:, 1].round(4)
    out["prob_away_win"]    = probs[:, 2].round(4)
    out["upset_risk"]       = (1 - np.max(probs, axis=1)).round(4)
    out["predicted_at_utc"] = snapshot_time.isoformat()
    match_dates = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["pre_match_snapshot"] = match_dates > snapshot_time.date()

    # Archetype-aware confidence note
    out["archetype_context"] = matches_2026.apply(
        lambda r: ",".join(
            col for col in ARCHETYPE_LABELS
            if col in matches_2026.columns and r.get(col, 0) == 1
        ) or "none",
        axis=1
    ).values

    return out


def validate_predictions(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    """Compare predictions to actual results."""
    merged = preds.merge(
        actuals[["match_id", "result", "home_score", "away_score"]],
        on="match_id", how="inner"
    )
    merged = merged.dropna(subset=["result"])
    if len(merged) == 0:
        return {"n_validated": 0}

    correct = (merged["pred_result"] == merged["result"]).mean()
    # Log loss
    result_map = {"H": 0, "D": 1, "A": 2}
    y_true = merged["result"].map(result_map).dropna().astype(int)
    prob_cols = ["prob_home_win", "prob_draw", "prob_away_win"]
    probs = merged.loc[y_true.index, prob_cols].values.clip(1e-6, 1 - 1e-6)

    from sklearn.metrics import log_loss, brier_score_loss
    ll = log_loss(y_true, probs)
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), probs[:, c])
        for c in range(3)
    ])

    # Upset detection
    merged["actual_upset"] = (
        (merged["result"] == "A") & (merged["prob_home_win"] > 0.5) |
        (merged["result"] == "H") & (merged["prob_away_win"] > 0.5)
    )
    upset_rate = merged["actual_upset"].mean()

    # Per-archetype accuracy
    arch_acc = {}
    for col in ARCHETYPE_LABELS:
        if col in merged.columns:
            sub = merged[merged[col] == 1]
            if len(sub) > 0:
                arch_acc[col] = round((sub["pred_result"] == sub["result"]).mean(), 4)

    summary = {
        "n_validated": len(merged),
        "accuracy": round(float(correct), 4),
        "log_loss": round(float(ll), 4),
        "brier_score": round(float(brier), 4),
        "upset_rate_actual": round(float(upset_rate), 4),
        "pre_match_snapshot_count": int(preds.get("pre_match_snapshot", pd.Series(dtype=bool)).fillna(False).sum()),
        "retrospective_snapshot_count": int((~preds.get("pre_match_snapshot", pd.Series([False] * len(preds))).fillna(False)).sum()),
        "accuracy_by_archetype": arch_acc,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary, merged


def main():
    print("=" * 60)
    print("  Phase 9: 2026 Out-of-Sample Validation")
    print("=" * 60)

    mf = load_all_data()

    # Split: train on pre-2026 WC, predict 2026
    historical = mf[(mf["is_world_cup"] == True) & (mf["match_year"] < 2026)].copy()
    matches_26 = mf[(mf["is_world_cup"] == True) & (mf["match_year"] == 2026)].copy()

    print(f"  Historical WC matches (train): {len(historical):,}")
    print(f"  2026 WC matches (predict)     : {len(matches_26):,}")

    if len(matches_26) == 0:
        print("  [WARN] No 2026 WC matches found in matchup features.")
        print("  Checking raw 2026 live data...")

        live = pd.read_csv(PROCESSED_DIR / "2026_live_matches.csv", low_memory=False)
        print(f"  Live data: {len(live)} rows, columns: {list(live.columns)}")
        matches_26 = live.copy()

    # Train model on all historical WC data
    print("\n  Training model on historical WC data...")
    feat_cols_present = [c for c in FEATURE_COLS if c in historical.columns]
    model, imputer, scaler, feat_cols = train_best_model(historical)

    # Generate predictions for 2026
    print(f"\n  Generating predictions for {len(matches_26)} matches...")
    feat_cols_26 = [c for c in feat_cols if c in matches_26.columns]
    missing_feats = [c for c in feat_cols if c not in matches_26.columns]
    if missing_feats:
        print(f"  [INFO] {len(missing_feats)} features missing in 2026 data "
              f"(will be imputed): {missing_feats[:5]}")
        for c in missing_feats:
            matches_26[c] = np.nan

    preds = generate_predictions(model, imputer, scaler, feat_cols, matches_26)

    # Save predictions. Rows are only pre-match predictions when
    # pre_match_snapshot is true.
    preds.to_csv(PROCESSED_DIR / "predictions_2026.csv", index=False)
    print(f"  Saved predictions_2026.csv ({len(preds)} matches, "
          f"timestamped {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M UTC')})")

    # Print upcoming matches
    future = preds[pd.to_datetime(preds["date"], errors="coerce") >
                   pd.Timestamp.now()]
    played = preds[pd.to_datetime(preds["date"], errors="coerce") <=
                   pd.Timestamp.now()]

    print(f"\n  Predictions summary:")
    print(f"    Already played: {len(played)}")
    print(f"    Still upcoming: {len(future)}")

    if len(future) > 0:
        print(f"\n  Upcoming 2026 WC predictions:")
        print(f"  {'Date':<12} {'Home':<20} {'Away':<20} {'Pred':>6} {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}")
        print(f"  {'-'*78}")
        for _, row in future.sort_values("date").iterrows():
            print(f"  {str(row['date'])[:10]:<12} {str(row['home_team']):<20} "
                  f"{str(row['away_team']):<20} {str(row['pred_result']):>6} "
                  f"{row['prob_home_win']:>6.3f} {row['prob_draw']:>6.3f} "
                  f"{row['prob_away_win']:>6.3f}")

    # Validate against played matches that have results
    print(f"\n  Validating predictions against actual results...")
    actuals = mf[(mf["is_world_cup"] == True) & (mf["match_year"] == 2026) &
                 (mf["result"].notna())].copy()

    if len(actuals) > 0:
        validation_result = validate_predictions(preds, actuals)
        if isinstance(validation_result, tuple):
            summary, merged = validation_result
        else:
            summary = validation_result
            merged = pd.DataFrame()

        print(f"  Validated against {summary['n_validated']} completed 2026 matches")
        if summary["n_validated"] > 0:
            print(f"    Accuracy  : {summary['accuracy']:.1%}")
            print(f"    Log-loss  : {summary['log_loss']:.4f}")
            print(f"    Brier     : {summary['brier_score']:.4f}")
            print(f"    Upset rate: {summary['upset_rate_actual']:.1%}")

            if not merged.empty:
                merged.to_csv(PROCESSED_DIR / "validation_2026.csv", index=False)

        with open(PROCESSED_DIR / "validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved validation_summary.json")
    else:
        print(f"  [INFO] No completed 2026 WC matches found yet for validation.")
        summary = {"n_validated": 0,
                   "note": "Tournament in progress — validation pending final results"}
        with open(PROCESSED_DIR / "validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 9 complete ✓")
    print("=" * 60)

    return preds, summary


if __name__ == "__main__":
    main()
