"""
Phase 6: Rule-Based Archetype Classifier
==========================================
Assigns one or more archetype labels to each WC match using
explicit, threshold-based rules applied to fact_matchup_features.csv.

The 8 archetypes (multi-label, not mutually exclusive):
  1. heavyweight_clash       — both teams elite-rated
  2. favorite_vs_underdog    — large Elo gap
  3. host_pressure           — host nation involved
  4. generational_transition — unusual squad age profile
  5. club_power_mismatch     — large gap in elite-club representation
  6. tactical_contrast       — large gap in playing style (modern era only)
  7. knockout_volatility     — high-stakes elimination, close ratings
  8. upset_realized          — (post-hoc) lower-rated team actually won

Outputs:
  data/processed/fact_matchup_archetype.csv
    one row per WC match, columns: match_id + 8 archetype binary flags
    + archetype_label (comma-separated list) + feature context
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import PROCESSED_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLDS  (all sensitivity-analyzed in ±30% band)
# ─────────────────────────────────────────────────────────────────────────────

T = {
    # Heavyweight clash: both teams in top 25% of Elo ratings present in dataset
    "heavyweight_elo_pct":      75,    # percentile cutoff (computed from data)

    # Favorite vs. underdog: Elo gap exceeds this value
    "fav_underdog_elo_gap":     150,   # points

    # Generational transition: squad mean age is unusually young or old
    "gen_trans_age_low":        24.5,  # younger than this → young squad
    "gen_trans_age_high":       29.5,  # older than this → veteran squad
    # OR: one team is notably younger/older than the other
    "gen_trans_age_gap":        3.0,   # years difference between squads

    # Club-power mismatch: gap in top-5-league player share
    "club_mismatch_top5_gap":   0.25,  # e.g. 0.70 vs 0.45

    # Knockout volatility: elimination match with close ratings
    "ko_volatility_elo_gap":    80,    # Elo gap below this = "close"

    # Upset realized: lower-rated team won (simple binary)
    # No threshold needed — just check if result flipped expectation
}

ARCHETYPE_NAMES = [
    "heavyweight_clash",
    "favorite_vs_underdog",
    "host_pressure",
    "generational_transition",
    "club_power_mismatch",
    "tactical_contrast",
    "knockout_volatility",
    "upset_realized",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HOST NATION LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

# Historical WC host nations (one or more per edition)
WC_HOSTS = {
    1930: ["Uruguay"],
    1934: ["Italy"],
    1938: ["France"],
    1950: ["Brazil"],
    1954: ["Switzerland"],
    1958: ["Sweden"],
    1962: ["Chile"],
    1966: ["England"],
    1970: ["Mexico"],
    1974: ["West Germany"],
    1978: ["Argentina"],
    1982: ["Spain"],
    1986: ["Mexico"],
    1990: ["Italy"],
    1994: ["United States"],
    1998: ["France"],
    2002: ["South Korea", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["United States", "Canada", "Mexico"],
}


def is_host(team: str, year: int) -> bool:
    hosts = WC_HOSTS.get(int(year), [])
    t = str(team).lower()
    return any(h.lower() in t or t in h.lower() for h in hosts)


# ─────────────────────────────────────────────────────────────────────────────
#  RULE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rule_heavyweight(row, elo_p75: float) -> int:
    """Both teams in top Elo quartile of all WC teams in dataset."""
    eh = row.get("elo_home_pre", np.nan)
    ea = row.get("elo_away_pre", np.nan)
    if pd.isna(eh) or pd.isna(ea):
        return 0
    return int(eh >= elo_p75 and ea >= elo_p75)


def rule_fav_underdog(row) -> int:
    """Large Elo gap — one team is heavily favored."""
    gap = abs(row.get("elo_gap_abs", 0) or 0)
    return int(gap >= T["fav_underdog_elo_gap"])


def rule_host_pressure(row) -> int:
    """Host nation is playing in this match."""
    year = row.get("match_year") or row.get("date", "")[:4]
    try:
        year = int(year)
    except (TypeError, ValueError):
        return 0
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    return int(is_host(home, year) or is_host(away, year))


def rule_generational_transition(row) -> int:
    """One or both squads have unusual age profile (very young or very old)."""
    h_age = row.get("h_age_mean", np.nan)
    a_age = row.get("a_age_mean", np.nan)
    age_gap = abs(row.get("age_mean_gap_abs", np.nan) or np.nan)

    young_squad = (
        (not pd.isna(h_age) and h_age < T["gen_trans_age_low"]) or
        (not pd.isna(a_age) and a_age < T["gen_trans_age_low"])
    )
    old_squad = (
        (not pd.isna(h_age) and h_age > T["gen_trans_age_high"]) or
        (not pd.isna(a_age) and a_age > T["gen_trans_age_high"])
    )
    big_age_gap = not pd.isna(age_gap) and age_gap >= T["gen_trans_age_gap"]

    return int(young_squad or old_squad or big_age_gap)


def rule_club_mismatch(row) -> int:
    """Large gap in elite-club (top-5 league) player share."""
    gap = abs(row.get("top5_share_gap_abs", np.nan) or np.nan)
    if pd.isna(gap):
        return 0
    return int(gap >= T["club_mismatch_top5_gap"])


def rule_tactical_contrast(row) -> int:
    """
    Large gap in playing style. Currently proxied by league diversity gap
    (teams with high entropy = many leagues = diverse tactical input;
    teams with low entropy = one dominant league = coherent system).
    Full xG/possession features available 2014+; flag as unavailable before.
    """
    div_gap = abs(row.get("league_div_gap", np.nan) or np.nan)
    year = row.get("match_year", 0) or 0
    if pd.isna(div_gap) or year < 2006:
        return 0  # insufficient data to classify
    # High league diversity gap ≈ tactical contrast (proxy)
    return int(div_gap >= 0.5)


def rule_knockout_volatility(row) -> int:
    """Elimination match with closely matched teams."""
    is_ko = int(row.get("is_knockout", 0) or 0)
    gap = abs(row.get("elo_gap_abs", np.nan) or np.nan)
    if not is_ko or pd.isna(gap):
        return 0
    return int(gap < T["ko_volatility_elo_gap"])


def rule_upset_realized(row) -> int:
    """Post-hoc: lower-rated team won (only meaningful as outcome variable)."""
    result = str(row.get("result", "")).upper()
    gap = row.get("elo_gap", np.nan)
    if pd.isna(gap) or result == "D":
        return 0
    # Positive gap means home team rated higher
    home_favored = gap > 0
    home_won = result == "H"
    if home_favored and not home_won and result == "A":
        return 1  # away upset
    if not home_favored and home_won:
        return 1  # home upset
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  APPLY RULES
# ─────────────────────────────────────────────────────────────────────────────

def apply_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 8 rules to each row."""
    # Compute Elo percentile threshold from WC matches
    wc_elos = pd.concat([
        df["elo_home_pre"].dropna(),
        df["elo_away_pre"].dropna(),
    ])
    elo_p75 = float(wc_elos.quantile(T["heavyweight_elo_pct"] / 100))
    print(f"  Heavyweight Elo threshold (p{T['heavyweight_elo_pct']}): {elo_p75:.0f}")

    rules = {
        "heavyweight_clash":       lambda r: rule_heavyweight(r, elo_p75),
        "favorite_vs_underdog":    rule_fav_underdog,
        "host_pressure":           rule_host_pressure,
        "generational_transition": rule_generational_transition,
        "club_power_mismatch":     rule_club_mismatch,
        "tactical_contrast":       rule_tactical_contrast,
        "knockout_volatility":     rule_knockout_volatility,
        "upset_realized":          rule_upset_realized,
    }

    out = df.copy()
    for name, fn in rules.items():
        out[name] = out.apply(fn, axis=1).astype(int)

    # Multi-label string
    out["archetype_labels"] = out.apply(
        lambda r: ",".join(k for k in ARCHETYPE_NAMES if r.get(k, 0) == 1) or "none",
        axis=1
    )
    out["n_archetypes"] = out[ARCHETYPE_NAMES].sum(axis=1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SENSITIVITY ANALYSIS  (±30% threshold range)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(wc: pd.DataFrame) -> pd.DataFrame:
    """
    Vary each key threshold ±10%, ±20%, ±30% and report how archetype
    prevalence changes. Returns summary DataFrame.
    """
    results = []
    base_thresholds = {
        "fav_underdog_elo_gap": T["fav_underdog_elo_gap"],
        "ko_volatility_elo_gap": T["ko_volatility_elo_gap"],
        "club_mismatch_top5_gap": T["club_mismatch_top5_gap"],
        "gen_trans_age_gap": T["gen_trans_age_gap"],
    }

    for thresh_name, base_val in base_thresholds.items():
        for pct in [-30, -20, -10, 0, 10, 20, 30]:
            # Temporarily adjust threshold
            old_val = T[thresh_name]
            T[thresh_name] = base_val * (1 + pct / 100)

            reapplied = apply_archetypes(wc)

            for arch in ARCHETYPE_NAMES:
                if arch in reapplied.columns:
                    results.append({
                        "threshold": thresh_name,
                        "base_value": base_val,
                        "pct_change": pct,
                        "archetype": arch,
                        "prevalence": reapplied[arch].mean().round(4),
                        "n_matches": reapplied[arch].sum(),
                    })

            T[thresh_name] = old_val  # restore

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 6: Rule-Based Archetype Classifier")
    print("=" * 60)

    mf = pd.read_csv(PROCESSED_DIR / "fact_matchup_features.csv", low_memory=False)
    print(f"  Loaded matchup features: {len(mf):,} rows")

    # Apply to ALL matches (for later comparison) but focus QA on WC
    print("\n  Applying archetype rules...")
    result = apply_archetypes(mf)

    wc = result[result["is_world_cup"] == True].copy()
    print(f"\n  World Cup matches: {len(wc):,}")

    # ── Archetype prevalence ──────────────────────────────────────────────────
    print("\n  Archetype prevalence (WC matches):")
    print(f"  {'Archetype':<28} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*42}")
    for arch in ARCHETYPE_NAMES:
        if arch in wc.columns:
            n = int(wc[arch].sum())
            pct = wc[arch].mean() * 100
            flag = " ✓" if n >= 30 else " ⚠ (<30 instances)"
            print(f"  {arch:<28} {n:>6}  {pct:>5.1f}%{flag}")

    print(f"\n  Matches with ≥1 archetype: {(wc['n_archetypes'] > 0).sum():,}")
    print(f"  Avg archetypes per match  : {wc['n_archetypes'].mean():.2f}")

    # ── Top co-occurrences ───────────────────────────────────────────────────
    print("\n  Top archetype label combinations (WC):")
    label_counts = wc["archetype_labels"].value_counts().head(10)
    for label, cnt in label_counts.items():
        print(f"    {cnt:>4}x  {label}")

    # ── Era breakdown ────────────────────────────────────────────────────────
    wc["era"] = pd.cut(
        wc["match_year"],
        bins=[0, 1969, 1998, 2013, 2026],
        labels=["pre-1970", "1970–1998", "2002–2013", "2014–2026"]
    )
    print("\n  Archetype coverage by era (% of matches with ≥1 archetype):")
    era_cov = wc.groupby("era")["n_archetypes"].apply(lambda x: (x > 0).mean())
    for era, cov in era_cov.items():
        print(f"    {str(era):<14}  {cov:.1%}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_cols = (
        ["match_id", "date", "match_year", "tournament", "stage", "is_world_cup",
         "home_team", "away_team", "result",
         "elo_home_pre", "elo_away_pre", "elo_gap", "elo_gap_abs",
         "win_prob_home", "is_knockout", "is_neutral"]
        + ARCHETYPE_NAMES
        + ["archetype_labels", "n_archetypes"]
    )
    out_cols = [c for c in out_cols if c in result.columns]
    result[out_cols].to_csv(PROCESSED_DIR / "fact_matchup_archetype.csv", index=False)
    print(f"\n  Saved → fact_matchup_archetype.csv ({len(result):,} rows)")

    # ── Sensitivity analysis ─────────────────────────────────────────────────
    print("\n  Running sensitivity analysis (±30% threshold range)...")
    sens = sensitivity_analysis(wc)
    sens.to_csv(PROCESSED_DIR / "archetype_sensitivity.csv", index=False)

    # Quick summary: max prevalence swing per threshold
    print("\n  Sensitivity summary (max prevalence swing by threshold):")
    for thresh in sens["threshold"].unique():
        sub = sens[sens["threshold"] == thresh]
        swing = sub.groupby("archetype")["prevalence"].apply(lambda x: x.max() - x.min())
        top_arch = swing.idxmax()
        print(f"    {thresh:<30} most sensitive: {top_arch} (±{swing.max():.1%})")

    print("\n" + "=" * 60)
    print("  Phase 6 complete ✓")
    print("=" * 60)

    return result, wc


if __name__ == "__main__":
    main()
