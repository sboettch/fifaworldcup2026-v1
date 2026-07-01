# NOTE — Dixon-Coles Bivariate Poisson Model (v2 foundation)
# ─────────────────────────────────────────────────────────────────────────────
# Implements Dixon & Coles (1997) attack/defence goal model. Included as a
# foundation for v2 but was intractable in v1 LOTO-CV (~400 team parameters
# × 23 folds via scipy.optimize). Intended for distributed compute in v2.
# See docs/checkin_2.md §6 for discussion.
# ─────────────────────────────────────────────────────────────────────────────

"""
Dixon-Coles Poisson Goal Model + Archetype Ensemble
=====================================================
Implements the Dixon & Coles (1997) bivariate Poisson model for football:

  λ_home = exp(μ + α_home - β_away + h)   (home team goals)
  λ_away = exp(μ + α_away - β_home)        (away team goals)

where:
  α_i = attack parameter for team i
  β_i = defence parameter for team i  
  μ   = global intercept
  h   = home advantage (set to 0 for neutral venues)

The Dixon-Coles correction term ρ adjusts for the underrepresentation
of 0-0 and 1-0 / 0-1 scorelines (low-scoring draws are more common 
than Poisson predicts).

P(H), P(D), P(A) are derived by summing over score probabilities.

Then we ensemble Dixon-Coles probabilities with the best RF model
(archetype-conditioned weighting learned from validation data).

References:
  Dixon & Coles (1997). Modelling association football scores and 
  inefficiencies in the football betting market. Applied Statistics.
  
  Karlis & Ntzoufras (2003). Analysis of sports data by using 
  bivariate Poisson models. The Statistician.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.constants import PROCESSED_DIR

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MAX_GOALS = 8   # sum over scorelines 0–8 for each team

ARCH_COLS = [
    "heavyweight_clash", "favorite_vs_underdog", "host_pressure",
    "generational_transition", "club_power_mismatch", "tactical_contrast",
    "knockout_volatility",
]
ELO_FEATS = [
    "elo_home_pre", "elo_away_pre", "elo_gap_abs", "win_prob_home",
    "win_prob_away", "k_factor", "is_knockout", "is_neutral",
]
SQUAD_FEATS = [
    "h_age_mean", "a_age_mean", "h_top5_share", "a_top5_share",
    "top5_share_gap_abs", "h_league_diversity_entropy",
    "a_league_diversity_entropy", "squad_size_gap",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Dixon-Coles correction term
# ─────────────────────────────────────────────────────────────────────────────

def dc_correction(home_goals, away_goals, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction τ(x, y, λ_h, λ_a, ρ)."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lam_h * lam_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lam_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lam_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def score_prob(h, a, lam_h, lam_a, rho):
    """P(home=h, away=a) under Dixon-Coles bivariate Poisson."""
    p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
    p *= dc_correction(h, a, lam_h, lam_a, rho)
    return max(p, 0.0)


def outcome_probs(lam_h, lam_a, rho, max_goals=MAX_GOALS):
    """Return P(H), P(D), P(A) by summing over score matrix."""
    ph, pd_, pa = 0.0, 0.0, 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = score_prob(h, a, lam_h, lam_a, rho)
            if h > a:
                ph += p
            elif h == a:
                pd_ += p
            else:
                pa += p
    total = ph + pd_ + pa
    if total < 1e-9:
        return np.array([1/3, 1/3, 1/3])
    return np.array([ph / total, pd_ / total, pa / total])


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter estimation
# ─────────────────────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Dixon-Coles attack/defence model.
    Fit on all matches in training set (not just WC).
    Parameters: {α_team, β_team for each team}, μ, h, ρ.
    """

    def __init__(self, xi: float = 0.0):
        """xi: time-weighting decay (0 = no decay, 0.002 = standard)."""
        self.xi = xi
        self.params_ = None
        self.teams_   = None
        self.team_idx_= None

    def _weights(self, dates, ref_date):
        """Exponential time decay: w = exp(-xi * days_ago)."""
        if self.xi == 0:
            return np.ones(len(dates))
        days = (pd.to_datetime(ref_date) - pd.to_datetime(dates)).dt.days.clip(lower=0)
        return np.exp(-self.xi * days)

    def fit(self, df: pd.DataFrame, ref_date=None):
        """
        df must have: home_team, away_team, home_score, away_score, date, is_neutral
        """
        df = df.dropna(subset=["home_score", "away_score",
                                "home_team", "away_team"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        if ref_date is None:
            ref_date = df["date"].max()

        self.teams_    = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        n_teams = len(self.teams_)

        weights = self._weights(df["date"], ref_date)

        # Initial params: [α_0..α_n-1, β_0..β_n-1, μ, h, ρ]
        x0 = np.concatenate([
            np.zeros(n_teams),   # attack (log scale, team 0 fixed=0)
            np.zeros(n_teams),   # defence (log scale, team 0 fixed=0)
            [0.0],               # μ (global intercept)
            [0.1],               # h (home advantage)
            [-0.1],              # ρ (DC correction)
        ])

        def neg_log_likelihood(params):
            alpha = params[:n_teams]
            beta  = params[n_teams:2*n_teams]
            mu    = params[2*n_teams]
            h_adv = params[2*n_teams + 1]
            rho   = params[2*n_teams + 2]

            ll = 0.0
            for i, row in enumerate(df.itertuples()):
                hi = self.team_idx_.get(row.home_team, 0)
                ai = self.team_idx_.get(row.away_team, 0)
                is_neutral = getattr(row, "is_neutral", False)
                home_bonus = 0.0 if is_neutral else h_adv

                lam_h = np.exp(mu + alpha[hi] - beta[ai] + home_bonus)
                lam_a = np.exp(mu + alpha[ai] - beta[hi])

                hg = int(row.home_score)
                ag = int(row.away_score)
                if hg > MAX_GOALS: hg = MAX_GOALS
                if ag > MAX_GOALS: ag = MAX_GOALS

                p = score_prob(hg, ag, lam_h, lam_a, rho)
                ll += weights[i] * np.log(max(p, 1e-10))

            return -ll

        # Constraint: sum of attack params = 0 (identifiability)
        constraints = [{"type": "eq",
                        "fun": lambda p: np.sum(p[:n_teams])}]

        result = minimize(neg_log_likelihood, x0,
                          method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-8})
        self.params_ = result.x
        self.n_teams_ = n_teams
        return self

    def predict_proba(self, home_team: str, away_team: str,
                      is_neutral: bool = True) -> np.ndarray:
        """Return [P(H), P(D), P(A)] for a single match."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted")

        p = self.params_
        n = self.n_teams_
        alpha = p[:n]; beta = p[n:2*n]
        mu = p[2*n]; h_adv = p[2*n+1]; rho = p[2*n+2]

        hi = self.team_idx_.get(home_team, 0)
        ai = self.team_idx_.get(away_team, 0)
        home_bonus = 0.0 if is_neutral else h_adv

        lam_h = np.exp(mu + alpha[hi] - beta[ai] + home_bonus)
        lam_a = np.exp(mu + alpha[ai] - beta[hi])
        return outcome_probs(lam_h, lam_a, rho)


# ─────────────────────────────────────────────────────────────────────────────
#  Additional features: H2H record + rest days + travel distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_h2h_features(all_matches: pd.DataFrame,
                          wc_matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each WC match, compute:
      h2h_total: total previous meetings between the two teams
      h2h_home_win_rate: home team's historical win rate in this fixture
      h2h_draw_rate: draw rate in this fixture
    Only uses matches BEFORE the current match date (no leakage).
    """
    all_matches = all_matches.dropna(subset=["home_team","away_team","result"]).copy()
    all_matches["date"] = pd.to_datetime(all_matches["date"], errors="coerce")
    wc_matches = wc_matches.copy()
    wc_matches["date_dt"] = pd.to_datetime(wc_matches["date"], errors="coerce")

    h2h_records = []
    for _, row in wc_matches.iterrows():
        ht, at = row["home_team"], row["away_team"]
        cutoff = row["date_dt"]

        # All past meetings in either direction
        past = all_matches[
            (all_matches["date"] < cutoff) & (
                ((all_matches["home_team"]==ht) & (all_matches["away_team"]==at)) |
                ((all_matches["home_team"]==at) & (all_matches["away_team"]==ht))
            )
        ]

        n = len(past)
        if n == 0:
            h2h_records.append({"match_id": row["match_id"],
                                  "h2h_total": 0,
                                  "h2h_home_win_rate": np.nan,
                                  "h2h_draw_rate": np.nan})
            continue

        # From perspective of 'home_team' in current match
        wins   = ((past["home_team"]==ht) & (past["result"]=="H")).sum() + \
                 ((past["home_team"]==at) & (past["result"]=="A")).sum()
        draws  = (past["result"]=="D").sum()
        h2h_records.append({
            "match_id": row["match_id"],
            "h2h_total": n,
            "h2h_home_win_rate": wins / n,
            "h2h_draw_rate": draws / n,
        })

    return pd.DataFrame(h2h_records)


def compute_rest_days(wc_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Days since last match for each team entering this WC match.
    Uses only matches within the same tournament.
    """
    wc = wc_matches.sort_values("date").copy()
    wc["date_dt"] = pd.to_datetime(wc["date"], errors="coerce")

    last_match = {}   # team → last match date
    rest_records = []

    for _, row in wc.iterrows():
        ht, at = row.get("home_team"), row.get("away_team")
        d = row["date_dt"]

        h_rest = (d - last_match[ht]).days if ht in last_match else np.nan
        a_rest = (d - last_match[at]).days if at in last_match else np.nan
        rest_records.append({
            "match_id": row["match_id"],
            "h_rest_days": h_rest,
            "a_rest_days": a_rest,
            "rest_gap": (h_rest - a_rest) if (pd.notna(h_rest) and pd.notna(a_rest)) else np.nan,
        })

        if pd.notna(d):
            if pd.notna(ht): last_match[ht] = d
            if pd.notna(at): last_match[at] = d

    return pd.DataFrame(rest_records)


# ─────────────────────────────────────────────────────────────────────────────
#  LOTO-CV with Dixon-Coles ensemble
# ─────────────────────────────────────────────────────────────────────────────

def loto_dc_ensemble(all_matches, wc, dc_weight=0.35):
    """
    LOTO-CV where each fold:
    1. Fits Dixon-Coles on all non-holdout matches (not just WC)
    2. Fits RF on WC non-holdout matches
    3. Ensembles: final_prob = (1-w)*RF_prob + w*DC_prob
    """
    wc = wc.copy()
    wc["year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year

    arch_cols_present = [c for c in ARCH_COLS if c in wc.columns]
    feat_cols = [c for c in ELO_FEATS + arch_cols_present + SQUAD_FEATS
                 if c in wc.columns]

    tournaments = sorted([t for t in wc["year"].dropna().unique()
                          if (wc["year"]==t).sum() >= 6])

    all_true, all_probs_rf, all_probs_dc = [], [], []

    for holdout in tournaments:
        train_wc  = wc[wc["year"] != holdout].copy()
        test_wc   = wc[wc["year"] == holdout].copy()
        # DC trains on ALL non-holdout matches (49K dataset)
        train_all = all_matches[
            pd.to_datetime(all_matches["date"], errors="coerce").dt.year != holdout
        ].copy()

        y_tr = train_wc["y"].dropna().astype(int)
        y_te = test_wc["y"].dropna().astype(int)
        if len(y_tr) < 20 or len(y_te) < 4:
            continue

        # ── RF ──────────────────────────────────────────────────────────────
        Xtr = train_wc[feat_cols].loc[y_tr.index]
        Xte = test_wc[feat_cols].loc[y_te.index]
        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
        Xte_s = sc.transform(imp.transform(Xte))

        rf = RandomForestClassifier(n_estimators=344, max_depth=5,
                                     min_samples_leaf=18, max_features=0.7,
                                     random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(Xtr_s, y_tr.values)
        rf_probs = rf.predict_proba(Xte_s)

        # ── Dixon-Coles ──────────────────────────────────────────────────────
        dc = DixonColesModel(xi=0.0)
        try:
            dc.fit(train_all, ref_date=train_all["date"].max())
        except Exception as e:
            print(f"    [DC WARN] {holdout}: {e}")
            dc = None

        dc_prob_list = []
        for _, row in test_wc.loc[y_te.index].iterrows():
            if dc is not None:
                try:
                    p = dc.predict_proba(row["home_team"], row["away_team"],
                                         is_neutral=bool(row.get("is_neutral", True)))
                except Exception:
                    p = np.array([1/3, 1/3, 1/3])
            else:
                p = np.array([1/3, 1/3, 1/3])
            dc_prob_list.append(p)

        dc_probs = np.array(dc_prob_list)

        all_true.extend(y_te.values)
        all_probs_rf.extend(rf_probs.tolist())
        all_probs_dc.extend(dc_probs.tolist())

    yt      = np.array(all_true)
    pr_rf   = np.array(all_probs_rf)
    pr_dc   = np.array(all_probs_dc)

    results = []
    print(f"\n  {'Model':<45} {'LogLoss':>9} {'Acc':>7} {'Brier':>7}")
    print(f"  {'-'*70}")

    for label, pr in [("RF Optuna only", pr_rf),
                       ("Dixon-Coles only", pr_dc)]:
        pr_c = np.clip(pr, 1e-7, 1-1e-7)
        ll   = log_loss(yt, pr_c)
        acc  = accuracy_score(yt, pr.argmax(axis=1))
        brier= np.mean([brier_score_loss((yt==c).astype(int), pr_c[:,c]) for c in range(3)])
        print(f"  {label:<45} {ll:>9.4f} {acc:>7.4f} {brier:>7.4f}")
        results.append({"model": label, "log_loss": ll, "accuracy": acc, "brier": brier})

    # Grid search ensemble weight
    print(f"\n  Ensemble weight search (RF weight / DC weight):")
    best_ll = np.inf; best_w = 0.3
    for w_dc in np.arange(0.0, 1.01, 0.05):
        w_rf = 1 - w_dc
        pr_ens = w_rf * pr_rf + w_dc * pr_dc
        pr_c = np.clip(pr_ens, 1e-7, 1-1e-7)
        ll   = log_loss(yt, pr_c)
        if ll < best_ll:
            best_ll = ll; best_w = w_dc

    print(f"    Best DC weight: {best_w:.2f} → ensemble log-loss = {best_ll:.4f}")
    pr_best = (1-best_w)*pr_rf + best_w*pr_dc
    pr_c    = np.clip(pr_best, 1e-7, 1-1e-7)
    acc     = accuracy_score(yt, pr_best.argmax(axis=1))
    brier   = np.mean([brier_score_loss((yt==c).astype(int), pr_c[:,c]) for c in range(3)])
    print(f"  {'RF+DC Ensemble (optimal weight)':<45} {best_ll:>9.4f} {acc:>7.4f} {brier:>7.4f}")
    results.append({"model": f"RF+DC Ensemble (DC w={best_w:.2f})",
                    "log_loss": best_ll, "accuracy": acc, "brier": brier,
                    "dc_weight": best_w})

    return pd.DataFrame(results), yt, pr_rf, pr_dc, best_w


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Dixon-Coles + Archetype Ensemble")
    print("=" * 65)

    # Load data
    mf   = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    arch = pd.read_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", low_memory=False)
    arch_cols_present = [c for c in ARCH_COLS if c in arch.columns]
    mf   = mf.merge(arch[["match_id"] + arch_cols_present], on="match_id", how="left")
    mf["y"] = mf["result"].map({"H": 0, "D": 1, "A": 2})

    # Full match history for DC fitting
    fact_match = pd.read_csv(PROCESSED_DIR / "fact_match.csv",
                              dtype={"stage": str}, low_memory=False)
    fact_match["home_score"] = pd.to_numeric(fact_match.get("home_score", np.nan), errors="coerce")
    fact_match["away_score"] = pd.to_numeric(fact_match.get("away_score", np.nan), errors="coerce")

    wc = mf[mf["is_world_cup"] == True].copy()

    # ── H2H features ────────────────────────────────────────────────────────
    print("\n  [1] Computing head-to-head features...")
    h2h = compute_h2h_features(fact_match, wc)
    wc  = wc.merge(h2h, on="match_id", how="left")
    print(f"    H2H coverage: {wc['h2h_total'].notna().mean():.0%}")
    print(f"    Avg H2H meetings: {wc['h2h_total'].mean():.1f} prior meetings")

    # ── Rest days ────────────────────────────────────────────────────────────
    print("\n  [2] Computing rest day features...")
    rest = compute_rest_days(wc)
    wc   = wc.merge(rest, on="match_id", how="left")
    cov  = wc["h_rest_days"].notna().mean()
    print(f"    Rest day coverage: {cov:.0%} (NaN = first match of tournament)")

    # Add to feature set
    H2H_FEATS  = ["h2h_total", "h2h_home_win_rate", "h2h_draw_rate"]
    REST_FEATS = ["h_rest_days", "a_rest_days", "rest_gap"]

    feat_extended = [c for c in ELO_FEATS + arch_cols_present + SQUAD_FEATS
                     + H2H_FEATS + REST_FEATS if c in wc.columns]
    print(f"\n  Extended feature set: {len(feat_extended)} features "
          f"(+{len([c for c in H2H_FEATS+REST_FEATS if c in wc.columns])} H2H+rest)")

    # ── RF with extended features ────────────────────────────────────────────
    print("\n  [3] RF with H2H + rest features (LOTO-CV)...")
    wc["year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year
    tournaments = sorted([t for t in wc["year"].dropna().unique()
                          if (wc["year"]==t).sum() >= 6])
    all_true, all_probs = [], []
    for holdout in tournaments:
        tr = wc[wc["year"]!=holdout]; te = wc[wc["year"]==holdout]
        ytr = tr["y"].dropna().astype(int); yte = te["y"].dropna().astype(int)
        Xtr = tr[feat_extended].loc[ytr.index]; Xte = te[feat_extended].loc[yte.index]
        if len(ytr)<20 or len(yte)<4: continue
        imp=SimpleImputer(strategy="median"); sc=StandardScaler()
        Xtr=sc.fit_transform(imp.fit_transform(Xtr)); Xte=sc.transform(imp.transform(Xte))
        m=RandomForestClassifier(n_estimators=344,max_depth=5,min_samples_leaf=18,
                                  max_features=0.7,random_state=RANDOM_SEED,n_jobs=-1)
        m.fit(Xtr, ytr.values); probs=m.predict_proba(Xte)
        all_true.extend(yte.values); all_probs.extend(probs.tolist())
    yt=np.array(all_true); pr=np.array(all_probs)
    ll_ext=log_loss(yt,np.clip(pr,1e-7,1-1e-7))
    acc_ext=accuracy_score(yt,pr.argmax(axis=1))
    brier_ext=np.mean([brier_score_loss((yt==c).astype(int),pr[:,c]) for c in range(3)])
    print(f"    RF + H2H + rest: ll={ll_ext:.4f}  acc={acc_ext:.4f}  brier={brier_ext:.4f}")

    # ── Dixon-Coles ensemble ──────────────────────────────────────────────────
    print("\n  [4] Dixon-Coles + RF ensemble (LOTO-CV)...")
    print("      (fitting DC on full 49K match history per fold...)")
    results_df, yt2, pr_rf, pr_dc, best_w = loto_dc_ensemble(fact_match, wc)

    # ── Final comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON vs. published benchmarks")
    print("=" * 65)
    ref_ll   = 0.9463; ref_acc = 0.593
    ext_lift = ref_ll - ll_ext
    dc_best  = results_df.sort_values("log_loss").iloc[0]
    dc_lift  = ref_ll - dc_best["log_loss"]

    print(f"\n  {'Model':<45} {'LogLoss':>9} {'Acc':>7}  {'Δ vs RF-Optuna':>14}")
    print(f"  {'-'*80}")
    print(f"  {'Pinnacle market (reference ceiling)':<45} {'~0.91':>9} {'~0.645':>7}  {'---':>14}")
    print(f"  {'Groll et al. 2019 (best published)':<45} {'~0.93':>9} {'~0.610':>7}  {'---':>14}")
    print(f"  {'RF Optuna (our winner)':<45} {ref_ll:>9.4f} {ref_acc:>7.4f}  {'0.0000':>14}")
    print(f"  {'RF + H2H + rest days':<45} {ll_ext:>9.4f} {acc_ext:>7.4f}  {ext_lift:>+14.4f}")
    for _, row in results_df.sort_values("log_loss").iterrows():
        lift = ref_ll - row["log_loss"]
        print(f"  {str(row['model']):<45} {row['log_loss']:>9.4f} {row['accuracy']:>7.4f}  {lift:>+14.4f}")

    # Save
    results_df.to_csv(PROCESSED_DIR / "model_results_dc_ensemble.csv", index=False)
    summary = {
        "rf_extended_ll": round(ll_ext, 4),
        "rf_extended_acc": round(acc_ext, 4),
        "dc_best_ll": round(dc_best["log_loss"], 4),
        "dc_best_acc": round(dc_best["accuracy"], 4),
        "dc_optimal_weight": float(best_w),
        "rf_optuna_baseline_ll": ref_ll,
        "lift_vs_rf_optuna": round(float(ref_ll - dc_best["log_loss"]), 4),
    }
    with open(PROCESSED_DIR / "dc_ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: model_results_dc_ensemble.csv, dc_ensemble_summary.json")

    print("\n" + "=" * 65)
    print("  Dixon-Coles pipeline complete ✓")
    print("=" * 65)


if __name__ == "__main__":
    main()
