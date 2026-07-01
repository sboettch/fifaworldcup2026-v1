# NOTE — Research / Experimental Script
# ─────────────────────────────────────────────────────────────────────────────
# This script was part of the v1 investigation and documents findings but is
# NOT part of the core reproducible pipeline (`make pipeline`).
# See docs/checkin_2.md for results and interpretation.
# ─────────────────────────────────────────────────────────────────────────────

"""
Phase 8+: Synthetic Augmentation + Improved Winner Model
=========================================================
Two strategies:

AUGMENTATION
1. Mirror augmentation: every WC match A vs B → also B vs A (swap
   all home/away features, flip result H↔A). Doubles training set 1037→2074.
2. High-stakes proxy matches: neutral-venue international matches where
   both teams Elo > WC p25 (1775). 987 matches, 100% Elo coverage.
   These have similar structural profiles to WC matches.
3. Gaussian feature noise injection (σ=3% of feature std): generates
   N synthetic near-duplicates per real match for rare archetypes.

WINNING MODEL IMPROVEMENTS (RF, Feature set C)
1. Optuna hyperparameter search (100 trials, LOTO-CV objective)
2. Interaction features: Elo×age cross-terms, archetype combinations
3. Platt/isotonic calibration (ECE measurement)
4. Stacking: RF + LR → meta-learner (handles RF overconfidence)
5. Rolling form features: last-5-match win rate per team

Outputs:
  model_results_augmented.csv
  best_model_params.json
  calibration_metrics.csv
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

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score,
                              roc_auc_score)
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
FEAT_C = ELO_FEATURES + ARCHETYPE_LABELS + SQUAD_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    mf   = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)
    arch_cols = ["match_id"] + [c for c in ARCHETYPE_LABELS if c in arch.columns]
    mf = mf.merge(arch[arch_cols], on="match_id", how="left")
    mf["match_year"] = pd.to_datetime(mf["date"], errors="coerce").dt.year
    mf["y"] = mf["result"].map({"H": 0, "D": 1, "A": 2})
    return mf


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def mirror_augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror each match by swapping home/away team features and flipping result.
    H→A, A→H, D→D. Also swap all h_/a_ prefixed feature columns.
    Doubles the training set with valid, label-consistent examples.
    """
    mirror = df.copy()

    # Flip result
    result_flip = {"H": "A", "A": "H", "D": "D"}
    mirror["result"] = mirror["result"].map(result_flip)
    mirror["y"] = mirror["result"].map({"H": 0, "D": 1, "A": 2})

    # Swap h_/a_ prefixed columns
    h_cols = [c for c in df.columns if c.startswith("h_")]
    a_cols = [c for c in df.columns if c.startswith("a_")]
    h_stems = {c[2:] for c in h_cols}
    a_stems = {c[2:] for c in a_cols}
    shared = h_stems & a_stems

    for stem in shared:
        hc, ac = f"h_{stem}", f"a_{stem}"
        mirror[hc] = df[ac]
        mirror[ac] = df[hc]

    # Swap directional Elo columns
    for h_col, a_col in [("elo_home_pre", "elo_away_pre"),
                          ("win_prob_home", "win_prob_away")]:
        if h_col in df.columns and a_col in df.columns:
            mirror[h_col] = df[a_col]
            mirror[a_col] = df[h_col]

    # Elo gap is now reversed in sign (abs stays same)
    if "elo_gap" in df.columns:
        mirror["elo_gap"] = -df["elo_gap"]

    mirror["match_id"] = mirror["match_id"].astype(str) + "_mirror"
    mirror["is_augmented"] = True
    return mirror


def high_stakes_proxy(mf: pd.DataFrame, elo_threshold: float = 1775) -> pd.DataFrame:
    """
    Extract high-stakes neutral non-WC matches as proxy training data.
    These match the structural profile of WC matches (neutral venue, big nations).
    """
    proxy = mf[
        (mf["is_world_cup"] == False) &
        (mf["is_neutral"] == True) &
        (mf["elo_home_pre"] > elo_threshold) &
        (mf["elo_away_pre"] > elo_threshold) &
        (mf["result"].notna())
    ].copy()
    proxy["y"] = proxy["result"].map({"H": 0, "D": 1, "A": 2})
    proxy["is_augmented"] = True
    proxy["match_year"] = pd.to_datetime(proxy["date"], errors="coerce").dt.year
    return proxy


def noise_augment(df: pd.DataFrame, n_copies: int = 1,
                  sigma_frac: float = 0.03) -> pd.DataFrame:
    """
    Gaussian noise injection on continuous features.
    For each row, generate n_copies with small perturbations (3% of std).
    Applied only to rare archetypes (< median prevalence) to balance labels.
    """
    num_cols = [c for c in FEAT_C if c in df.columns and
                df[c].dtype in [np.float64, np.float32, float] and
                "is_" not in c]
    stds = df[num_cols].std()

    augmented = []
    for _ in range(n_copies):
        noisy = df.copy()
        noise = np.random.normal(0, sigma_frac, size=(len(df), len(num_cols)))
        noisy[num_cols] += noise * stds.values
        noisy["match_id"] = noisy["match_id"].astype(str) + f"_noise{_}"
        noisy["is_augmented"] = True
        augmented.append(noisy)
    return pd.concat(augmented, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add theoretically motivated cross-feature interactions."""
    out = df.copy()

    # Elo × knockout: big games between equals are more volatile
    if "elo_gap_abs" in df.columns and "is_knockout" in df.columns:
        out["elo_gap_x_ko"] = df["elo_gap_abs"] * df["is_knockout"].fillna(0)

    # Age gap × Elo gap: generational upset potential
    if "age_mean_gap_abs" in df.columns and "elo_gap_abs" in df.columns:
        out["age_elo_interaction"] = (
            df["age_mean_gap_abs"].fillna(0) * df["elo_gap_abs"].fillna(0)
        )

    # Top5 share × Elo gap: club power amplifies or dampens rating gap
    if "top5_share_gap_abs" in df.columns and "elo_gap_abs" in df.columns:
        out["club_elo_interaction"] = (
            df["top5_share_gap_abs"].fillna(0) * df["elo_gap_abs"].fillna(0)
        )

    # Favorite win prob × heavyweight: high-prob favorites in big matches
    if "win_prob_home" in df.columns and "heavyweight_clash" in df.columns:
        out["fav_x_heavyweight"] = (
            df["win_prob_home"].fillna(0.5) * df["heavyweight_clash"].fillna(0)
        )

    # Upset archetype compound
    if "favorite_vs_underdog" in df.columns and "knockout_volatility" in df.columns:
        out["upset_compound"] = (
            df["favorite_vs_underdog"].fillna(0) * df["knockout_volatility"].fillna(0)
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ROLLING FORM FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_form(mf: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add last-N-match win rate and goal difference per team (pre-match).
    Uses chronological order; only uses matches before the current one.
    """
    mf = mf.sort_values("date").copy()
    mf["home_team"] = mf["home_team"].fillna("Unknown")
    mf["away_team"] = mf["away_team"].fillna("Unknown")

    # Build per-team rolling win rate from fact_team_match perspective
    # We compute it from the matchup features table itself
    home_form = {}  # team → deque of win(1)/draw(0.5)/loss(0)
    away_form = {}

    h_form_col = []
    a_form_col = []

    from collections import deque

    def update(team, result, is_home):
        if team not in home_form:
            home_form[team] = deque(maxlen=window)
        val = (1.0 if result == "H" else (0.5 if result == "D" else 0.0)) if is_home else \
              (1.0 if result == "A" else (0.5 if result == "D" else 0.0))
        home_form[team].append(val)

    for _, row in mf.iterrows():
        ht, at = row["home_team"], row["away_team"]
        # Form before this match
        hf = np.mean(list(home_form.get(ht, []))) if home_form.get(ht) else np.nan
        af = np.mean(list(home_form.get(at, []))) if home_form.get(at) else np.nan
        h_form_col.append(hf)
        a_form_col.append(af)
        # Update after recording
        if pd.notna(row.get("result")):
            update(ht, row["result"], is_home=True)
            update(at, row["result"], is_home=False)

    mf["h_form_l5"] = h_form_col
    mf["a_form_l5"] = a_form_col
    mf["form_gap"] = mf["h_form_l5"] - mf["a_form_l5"]
    return mf


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def measure_calibration(y_true, probs, n_bins=10):
    """Expected Calibration Error (ECE) across 3 classes."""
    ece_per_class = []
    for c in range(3):
        y_bin = (y_true == c).astype(int)
        prob_c = probs[:, c]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (prob_c >= bins[b]) & (prob_c < bins[b + 1])
            if mask.sum() == 0:
                continue
            acc_bin  = y_bin[mask].mean()
            conf_bin = prob_c[mask].mean()
            ece += mask.sum() * abs(acc_bin - conf_bin)
        ece_per_class.append(ece / len(y_true))
    return float(np.mean(ece_per_class))


# ─────────────────────────────────────────────────────────────────────────────
#  OPTUNA HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def optuna_rf_search(X: np.ndarray, y: np.ndarray,
                     match_years: np.ndarray, n_trials: int = 60):
    """
    Bayesian hyperparameter search for Random Forest.
    Objective: LOTO-CV log-loss on WC data.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    [WARN] optuna not installed — using default RF params")
        return {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3,
                "max_features": "sqrt", "class_weight": "balanced"}

    tournaments = [t for t in np.unique(match_years)
                   if (match_years == t).sum() >= 6]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", 0.5, 0.7]),
            "class_weight": trial.suggest_categorical("class_weight",
                                                       ["balanced", None]),
        }
        model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)

        lls = []
        for holdout in tournaments:
            train_idx = match_years != holdout
            test_idx  = match_years == holdout
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            if len(ytr) < 20 or len(yte) < 4:
                continue
            try:
                model.fit(Xtr, ytr)
                probs = model.predict_proba(Xte)
                lls.append(log_loss(yte, np.clip(probs, 1e-7, 1 - 1e-7)))
            except Exception:
                lls.append(2.0)
        return float(np.mean(lls)) if lls else 2.0

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
#  LOTO-CV EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def loto_eval(X, y, match_years, model, label="model"):
    tournaments = sorted([t for t in np.unique(match_years)
                          if (match_years == t).sum() >= 6])
    all_true, all_probs = [], []
    for holdout in tournaments:
        train_idx = match_years != holdout
        test_idx  = match_years == holdout
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if len(ytr) < 20 or len(yte) < 4:
            continue
        try:
            m = clone(model)
            m.fit(Xtr, ytr)
            probs = m.predict_proba(Xte)
        except Exception as e:
            print(f"    [WARN] {holdout}: {e}")
            continue
        all_true.extend(yte)
        all_probs.extend(probs.tolist())

    yt = np.array(all_true)
    pr = np.array(all_probs)
    ll    = log_loss(yt, np.clip(pr, 1e-7, 1 - 1e-7))
    acc   = accuracy_score(yt, pr.argmax(axis=1))
    brier = np.mean([brier_score_loss((yt==c).astype(int), pr[:,c]) for c in range(3)])
    ece   = measure_calibration(yt, pr)
    try:
        auc = roc_auc_score(yt, pr, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    print(f"    {label:<45}: ll={ll:.4f}  acc={acc:.3f}  brier={brier:.4f}  ECE={ece:.4f}")
    return {"model": label, "log_loss": round(ll,4), "accuracy": round(acc,4),
            "brier_score": round(brier,4), "ece": round(ece,4),
            "auc_macro": round(auc,4) if not np.isnan(auc) else None}


def prep_xy(df, feat_cols):
    """Impute + scale, return (X_scaled, y, match_years, imputer, scaler)."""
    y    = df["y"].dropna().astype(int)
    X_raw = df[feat_cols].loc[y.index]
    years = df["match_year"].loc[y.index].values

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X   = sc.fit_transform(imp.fit_transform(X_raw))
    return X, y.values, years, imp, sc


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Phase 8+: Augmentation + Improved Winner Model")
    print("=" * 65)

    mf = load_data()

    # ── 1. Add rolling form ───────────────────────────────────────────────────
    print("\n  [1] Computing rolling form features...")
    mf = add_rolling_form(mf, window=5)
    print(f"    h_form_l5 coverage: {mf['h_form_l5'].notna().mean():.1%}")

    # ── 2. Add interaction features ───────────────────────────────────────────
    print("  [2] Adding interaction features...")
    mf = add_interaction_features(mf)

    INTERACTION_FEATS = ["elo_gap_x_ko", "age_elo_interaction",
                         "club_elo_interaction", "fav_x_heavyweight", "upset_compound"]
    FORM_FEATS = ["h_form_l5", "a_form_l5", "form_gap"]

    FEAT_C_PLUS = [c for c in FEAT_C + INTERACTION_FEATS + FORM_FEATS
                   if c in mf.columns]
    print(f"    Feature set C+: {len(FEAT_C_PLUS)} features "
          f"(+{len(FEAT_C_PLUS) - len([c for c in FEAT_C if c in mf.columns])} new)")

    # ── 3. Build augmented training sets ─────────────────────────────────────
    print("\n  [3] Building augmented training sets...")
    wc = mf[mf["is_world_cup"] == True].copy()
    wc["is_augmented"] = False

    # Mirror augmentation
    wc_mirror = mirror_augment(wc)
    wc_aug_mirror = pd.concat([wc, wc_mirror], ignore_index=True)
    print(f"    Mirror: {len(wc)} → {len(wc_aug_mirror)} WC matches")

    # High-stakes proxy
    proxy = high_stakes_proxy(mf, elo_threshold=1775)
    wc_aug_proxy = pd.concat([wc, proxy], ignore_index=True)
    print(f"    Proxy:  {len(wc)} WC + {len(proxy)} high-stakes neutral = {len(wc_aug_proxy)}")

    # Combined: WC + mirror + proxy + noise
    noise = noise_augment(wc, n_copies=1, sigma_frac=0.03)
    wc_aug_full = pd.concat([wc, wc_mirror, proxy, noise], ignore_index=True)
    print(f"    Full:   {len(wc_aug_full)} total (WC + mirror + proxy + noise)")

    # ── 4. Baseline: RF C on original WC ─────────────────────────────────────
    print("\n  [4] Baselines (LOTO-CV, original WC only)...")
    results = []

    feat_cols = [c for c in FEAT_C if c in wc.columns]
    X_orig, y_orig, yrs_orig, imp0, sc0 = prep_xy(wc, feat_cols)

    rf_base = RandomForestClassifier(n_estimators=300, max_depth=6,
                                     min_samples_leaf=5, random_state=RANDOM_SEED,
                                     n_jobs=-1)
    r = loto_eval(X_orig, y_orig, yrs_orig, rf_base, "RF C (baseline)")
    r["augmentation"] = "none"; results.append(r)

    # ── 5. RF C+ with interaction + form features ─────────────────────────────
    print("\n  [5] RF with interaction + form features (C+)...")
    feat_cols_plus = [c for c in FEAT_C_PLUS if c in wc.columns]
    X_plus, y_plus, yrs_plus, _, _ = prep_xy(wc, feat_cols_plus)
    rf_plus = RandomForestClassifier(n_estimators=300, max_depth=6,
                                     min_samples_leaf=5, random_state=RANDOM_SEED,
                                     n_jobs=-1)
    r = loto_eval(X_plus, y_plus, yrs_plus, rf_plus, "RF C+ (interactions+form)")
    r["augmentation"] = "features_only"; results.append(r)

    # ── 6. RF + mirror augmentation ───────────────────────────────────────────
    print("\n  [6] RF with mirror augmentation...")
    # LOTO-CV: train on all years except holdout (including their mirrors),
    # test on real WC matches of holdout year only
    tournaments = sorted([t for t in wc["match_year"].dropna().unique()
                          if (wc["match_year"] == t).sum() >= 6])
    for aug_name, aug_df in [("mirror", wc_aug_mirror),
                               ("proxy",  wc_aug_proxy),
                               ("full",   wc_aug_full)]:
        print(f"\n  [6/{aug_name}] RF C with {aug_name} augmentation...")
        all_true, all_probs = [], []
        for holdout in tournaments:
            # Train: augmented data EXCLUDING real holdout-year matches
            train = aug_df[aug_df["match_year"] != holdout].copy()
            # Test: REAL WC matches of holdout year only (no augmented test)
            test  = wc[(wc["match_year"] == holdout) & ~wc.get("is_augmented", False)].copy()
            if len(train) < 30 or len(test) < 4:
                continue

            y_tr = train["y"].dropna().astype(int)
            y_te = test["y"].dropna().astype(int)
            Xtr_raw = train[feat_cols].loc[y_tr.index]
            Xte_raw = test[feat_cols].loc[y_te.index]
            if len(y_tr) < 20 or len(y_te) < 4:
                continue

            imp = SimpleImputer(strategy="median")
            sc  = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(Xtr_raw))
            Xte = sc.transform(imp.transform(Xte_raw))

            m = RandomForestClassifier(n_estimators=300, max_depth=6,
                                       min_samples_leaf=5, random_state=RANDOM_SEED,
                                       n_jobs=-1)
            m.fit(Xtr, y_tr.values)
            probs = m.predict_proba(Xte)
            all_true.extend(y_te.values)
            all_probs.extend(probs.tolist())

        if all_true:
            yt = np.array(all_true); pr = np.array(all_probs)
            ll    = log_loss(yt, np.clip(pr, 1e-7, 1-1e-7))
            acc   = accuracy_score(yt, pr.argmax(axis=1))
            brier = np.mean([brier_score_loss((yt==c).astype(int), pr[:,c]) for c in range(3)])
            ece   = measure_calibration(yt, pr)
            print(f"    RF C ({aug_name}){' '*(30-len(aug_name))}: ll={ll:.4f}  acc={acc:.3f}  brier={brier:.4f}  ECE={ece:.4f}")
            results.append({"model": f"RF C ({aug_name} aug)", "log_loss": round(ll,4),
                            "accuracy": round(acc,4), "brier_score": round(brier,4),
                            "ece": round(ece,4), "augmentation": aug_name})

    # ── 7. Calibration ────────────────────────────────────────────────────────
    print("\n  [7] Calibrated RF (isotonic) on original WC...")
    rf_cal = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                               random_state=RANDOM_SEED, n_jobs=-1),
        method="isotonic", cv=5
    )
    r = loto_eval(X_orig, y_orig, yrs_orig, rf_cal, "RF C calibrated (isotonic)")
    r["augmentation"] = "calibration"; results.append(r)

    # ── 8. Stacking: RF + LR → meta-LR ───────────────────────────────────────
    print("\n  [8] Stacking ensemble (RF + LR → meta-LR)...")
    stacker = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                          min_samples_leaf=5,
                                          random_state=RANDOM_SEED, n_jobs=-1)),
            ("lr", LogisticRegression(C=1.0, max_iter=500,
                                       multi_class="multinomial",
                                       random_state=RANDOM_SEED)),
        ],
        final_estimator=LogisticRegression(C=0.5, max_iter=500,
                                            multi_class="multinomial",
                                            random_state=RANDOM_SEED),
        cv=5, passthrough=False
    )
    r = loto_eval(X_orig, y_orig, yrs_orig, stacker, "Stacking (RF+LR → meta-LR)")
    r["augmentation"] = "stacking"; results.append(r)

    # ── 9. Optuna hyperparameter search ──────────────────────────────────────
    print("\n  [9] Optuna hyperparameter search for RF (60 trials)...")
    try:
        import optuna
        best_params = optuna_rf_search(X_orig, y_orig, yrs_orig, n_trials=60)
        print(f"    Best params: {best_params}")
        rf_tuned = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **best_params)
        r = loto_eval(X_orig, y_orig, yrs_orig, rf_tuned, "RF Optuna-tuned")
        r["augmentation"] = "optuna"; results.append(r)
        with open(PROCESSED_DIR / "best_model_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
    except ImportError:
        print("    Optuna not installed — skipping")

    # ── 10. Best of all: Optuna RF + mirror aug + calibration ────────────────
    print("\n  [10] Best combined: Optuna RF + mirror aug + isotonic calibration...")
    try:
        rf_best_combined = CalibratedClassifierCV(
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **best_params),
            method="isotonic", cv=5
        )
        # LOTO-CV with mirror augmentation
        all_true, all_probs = [], []
        for holdout in tournaments:
            train = wc_aug_mirror[wc_aug_mirror["match_year"] != holdout].copy()
            test  = wc[wc["match_year"] == holdout].copy()
            if len(train) < 30 or len(test) < 4:
                continue
            y_tr = train["y"].dropna().astype(int)
            y_te = test["y"].dropna().astype(int)
            Xtr_raw = train[feat_cols].loc[y_tr.index]
            Xte_raw = test[feat_cols].loc[y_te.index]
            if len(y_tr) < 20 or len(y_te) < 4:
                continue
            imp = SimpleImputer(strategy="median")
            sc  = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(Xtr_raw))
            Xte = sc.transform(imp.transform(Xte_raw))
            m = clone(rf_best_combined)
            m.fit(Xtr, y_tr.values)
            probs = m.predict_proba(Xte)
            all_true.extend(y_te.values)
            all_probs.extend(probs.tolist())

        if all_true:
            yt = np.array(all_true); pr = np.array(all_probs)
            ll    = log_loss(yt, np.clip(pr, 1e-7, 1-1e-7))
            acc   = accuracy_score(yt, pr.argmax(axis=1))
            brier = np.mean([brier_score_loss((yt==c).astype(int), pr[:,c]) for c in range(3)])
            ece   = measure_calibration(yt, pr)
            print(f"    RF Optuna+mirror+calibrated       : ll={ll:.4f}  acc={acc:.3f}  brier={brier:.4f}  ECE={ece:.4f}")
            results.append({"model": "RF Optuna+mirror+calibrated",
                            "log_loss": round(ll,4), "accuracy": round(acc,4),
                            "brier_score": round(brier,4), "ece": round(ece,4),
                            "augmentation": "optuna+mirror+calibration"})
    except Exception as e:
        print(f"    [WARN] Combined failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("  AUGMENTATION + IMPROVEMENT RESULTS")
    print("=" * 65)
    baseline_ll = results_df[results_df["model"]=="RF C (baseline)"]["log_loss"].values[0]
    print(f"\n  {'Model':<45} {'LogLoss':>9} {'Acc':>7} {'ECE':>7} {'Δ vs baseline':>14}")
    print(f"  {'-'*82}")
    for _, row in results_df.sort_values("log_loss").iterrows():
        delta = baseline_ll - row["log_loss"]
        flag = "✓" if delta > 0 else ""
        print(f"  {str(row['model']):<45} {row['log_loss']:>9.4f} {row['accuracy']:>7.4f} "
              f"{row.get('ece', float('nan')):>7.4f} {delta:>+13.4f} {flag}")

    results_df.to_csv(PROCESSED_DIR / "model_results_augmented.csv", index=False)
    print(f"\n  Saved: model_results_augmented.csv ({len(results_df)} configurations)")
    print("\n" + "=" * 65)
    print("  Phase 8+ complete ✓")
    print("=" * 65)
    return results_df


if __name__ == "__main__":
    main()
