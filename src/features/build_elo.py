"""
Phase 5: Elo Rating Engine
===========================
Rebuilds historical Elo ratings for all 342 teams from every match in
fact_match.csv (49,493 matches, 1872-2026) and generates:

  - fact_team_rating_snapshot.csv  : Elo rating per team after each match
  - fact_match_expectation.csv     : Pre-match Elo ratings + win probabilities
                                     for every match

Algorithm
---------
Standard Elo with these modifications for international football:
  - K-factor varies by match importance (WC > qualifier > friendly)
  - Home advantage: +100 Elo points added to home team's effective rating
    when computing win probability for non-neutral venues
  - Initial rating: 1500 for all teams on first appearance
  - Margin-of-victory multiplier (Hvattum & Arntzen 2010 inspired):
    K is scaled by ln(|GD| + 1) to reward convincing wins
  - New-team uncertainty: first 10 matches use K*1.5

References
----------
Hvattum, L.M. & Arntzen, H. (2010). Using ELO Ratings for Match Result
  Prediction in Association Football. IJF 26(3), 460-470.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import PROCESSED_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  ELO PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0   # effective Elo bonus for home team at non-neutral venues

# K-factors by match type
K_FACTORS = {
    "world_cup_final":    60,
    "world_cup_sf":       55,
    "world_cup_qf":       50,
    "world_cup_r16":      48,
    "world_cup_group":    45,
    "world_cup_other":    40,
    "confederation":      35,
    "qualifier":          25,
    "friendly":           15,
}

NEW_TEAM_MATCHES = 10     # first N matches: K *= 1.5
MOV_MULTIPLIER = True     # margin-of-victory scaling


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A vs team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def outcome_score(result: str, perspective: str) -> float:
    """Actual score (1=win, 0.5=draw, 0=loss) from home/away perspective."""
    if result == "D":
        return 0.5
    if perspective == "home":
        return 1.0 if result == "H" else 0.0
    else:  # away
        return 1.0 if result == "A" else 0.0


def get_k(row: pd.Series, match_count: dict) -> float:
    """Determine K-factor for this match."""
    is_wc = bool(row.get("is_world_cup", False))
    stage = str(row.get("stage", "")).lower()
    tournament = str(row.get("tournament", "")).lower()

    if is_wc or "world cup" in tournament:
        if "final" in stage and "semi" not in stage:
            k = K_FACTORS["world_cup_final"]
        elif "semi" in stage:
            k = K_FACTORS["world_cup_sf"]
        elif "quarter" in stage:
            k = K_FACTORS["world_cup_qf"]
        elif "round of 16" in stage or "round of 32" in stage:
            k = K_FACTORS["world_cup_r16"]
        elif "group" in stage:
            k = K_FACTORS["world_cup_group"]
        else:
            k = K_FACTORS["world_cup_other"]
    elif any(x in tournament for x in ["qualifier", "qualification", "eliminatoria"]):
        k = K_FACTORS["qualifier"]
    elif any(x in tournament for x in [
        "copa america", "euros", "euro", "afcon", "africa cup",
        "asian cup", "gold cup", "nations league", "confederation"
    ]):
        k = K_FACTORS["confederation"]
    else:
        k = K_FACTORS["friendly"]

    # New-team bonus
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    if match_count.get(home, 0) < NEW_TEAM_MATCHES:
        k *= 1.5
    if match_count.get(away, 0) < NEW_TEAM_MATCHES:
        k *= 1.5

    return k


def mov_multiplier(goal_diff: float) -> float:
    """Margin-of-victory multiplier (log scale, Hvattum & Arntzen)."""
    if not MOV_MULTIPLIER or pd.isna(goal_diff):
        return 1.0
    return np.log(abs(goal_diff) + 1.0) + 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  ELO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all matches chronologically and compute Elo ratings.

    Returns:
        snapshots  : one row per match per team (rating before & after)
        expectations: one row per match (pre-match ratings, probabilities)
    """
    # Sort chronologically
    matches = matches.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}    # team → current Elo rating
    match_count: dict[str, int] = {}  # team → number of matches played

    snapshot_rows = []
    expectation_rows = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if pd.isna(home) or pd.isna(away):
            continue

        # Starting ratings
        r_home = ratings.get(home, INITIAL_RATING)
        r_away = ratings.get(away, INITIAL_RATING)

        # Home advantage (not applied at neutral venues)
        neutral = bool(row.get("neutral", False))
        r_home_eff = r_home + (0 if neutral else HOME_ADVANTAGE)

        # Expected scores
        e_home = expected_score(r_home_eff, r_away)
        e_away = 1.0 - e_home

        # Actual result
        result = str(row.get("result", "D"))
        s_home = outcome_score(result, "home")
        s_away = outcome_score(result, "away")

        # Goal difference for MoV multiplier
        try:
            gd = float(row.get("home_score", 0) or 0) - float(row.get("away_score", 0) or 0)
        except (TypeError, ValueError):
            gd = 0.0

        k = get_k(row, match_count)
        mov = mov_multiplier(gd)

        # Rating updates
        delta_home = k * mov * (s_home - e_home)
        delta_away = k * mov * (s_away - e_away)

        r_home_new = r_home + delta_home
        r_away_new = r_away + delta_away

        # Win probability for home team (pre-match, without MoV)
        win_prob_home = e_home
        win_prob_away = e_away

        # Record expectation row
        expectation_rows.append({
            "match_id":         row.get("match_id"),
            "date":             row.get("date"),
            "home_team":        home,
            "away_team":        away,
            "is_world_cup":     row.get("is_world_cup", False),
            "stage":            row.get("stage"),
            "neutral":          neutral,
            "elo_home_pre":     round(r_home, 2),
            "elo_away_pre":     round(r_away, 2),
            "elo_gap":          round(r_home - r_away, 2),
            "elo_gap_abs":      round(abs(r_home - r_away), 2),
            "win_prob_home":    round(win_prob_home, 4),
            "win_prob_away":    round(win_prob_away, 4),
            "win_prob_draw":    round(1 - win_prob_home - win_prob_away + 2 * win_prob_home * win_prob_away, 4),
            "k_factor":         round(k, 2),
            "result":           result,
            "home_score":       row.get("home_score"),
            "away_score":       row.get("away_score"),
            "elo_home_post":    round(r_home_new, 2),
            "elo_away_post":    round(r_away_new, 2),
        })

        # Record snapshot rows (one per team)
        for team, r_pre, r_post in [(home, r_home, r_home_new), (away, r_away, r_away_new)]:
            snapshot_rows.append({
                "match_id":     row.get("match_id"),
                "date":         row.get("date"),
                "team":         team,
                "elo_before":   round(r_pre, 2),
                "elo_after":    round(r_post, 2),
                "elo_delta":    round(r_post - r_pre, 2),
                "match_count":  match_count.get(team, 0) + 1,
            })

        # Update state
        ratings[home] = r_home_new
        ratings[away] = r_away_new
        match_count[home] = match_count.get(home, 0) + 1
        match_count[away] = match_count.get(away, 0) + 1

    snapshots = pd.DataFrame(snapshot_rows)
    expectations = pd.DataFrame(expectation_rows)

    # Add rank column per-date (expensive but useful)
    # We compute end-of-tournament rankings by snapshotting at key dates
    final_ratings = pd.DataFrame([
        {"team": t, "elo_final": r} for t, r in ratings.items()
    ]).sort_values("elo_final", ascending=False).reset_index(drop=True)
    final_ratings["elo_rank"] = final_ratings.index + 1

    return snapshots, expectations, final_ratings


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_elo(expectations: pd.DataFrame, final_ratings: pd.DataFrame):
    """Quick sanity checks on Elo outputs."""
    print("\n  Validation:")

    # 1. Rating range
    wc = expectations[expectations["is_world_cup"] == True]
    print(f"    WC match expectations : {len(wc):,}")
    print(f"    Elo range (all teams) : "
          f"{final_ratings['elo_final'].min():.0f} – "
          f"{final_ratings['elo_final'].max():.0f}")

    # 2. Top 20 teams by final Elo
    print(f"\n  Top 20 teams by final Elo:")
    top = final_ratings.head(20)
    for _, r in top.iterrows():
        print(f"    #{int(r['elo_rank']):>3}  {r['team']:<30} {r['elo_final']:.0f}")

    # 3. Log-loss on WC matches as baseline quality check
    wc_valid = wc.dropna(subset=["result", "win_prob_home"])
    if len(wc_valid) > 0:
        # Actual outcome as 1/0
        y_home = (wc_valid["result"] == "H").astype(float)
        p_home = wc_valid["win_prob_home"].clip(1e-6, 1 - 1e-6)
        log_loss = -np.mean(
            y_home * np.log(p_home) + (1 - y_home) * np.log(1 - p_home)
        )
        print(f"\n  Binary log-loss on WC matches (Elo-only baseline): {log_loss:.4f}")
        print(f"  (Hvattum & Arntzen 2010 benchmark: ~0.65–0.70 for binary)")

    # 4. Calibration check: does a higher Elo team win more often?
    wc_valid = wc_valid.copy()
    wc_valid["predicted_winner"] = wc_valid["win_prob_home"] > 0.5
    wc_valid["home_won"] = wc_valid["result"] == "H"
    acc = (wc_valid["predicted_winner"] == wc_valid["home_won"]).mean()
    print(f"  Directional accuracy (predict home wins when p>0.5): {acc:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 5: Elo Rating Engine")
    print("=" * 60)

    fm = pd.read_csv(PROCESSED_DIR / "fact_match.csv", low_memory=False)
    print(f"  Loaded fact_match: {len(fm):,} matches")
    print(f"  Date range: {fm['date'].min()} – {fm['date'].max()}")

    # Only score matches where we have a result
    scored = fm[fm["result"].notna()].copy()
    print(f"  Scored matches: {len(scored):,}")

    print("\n  Running Elo engine...")
    snapshots, expectations, final_ratings = run_elo(scored)

    # Save outputs
    snapshots.to_csv(PROCESSED_DIR / "fact_team_rating_snapshot.csv", index=False)
    expectations.to_csv(PROCESSED_DIR / "fact_match_expectation.csv", index=False)
    final_ratings.to_csv(PROCESSED_DIR / "elo_final_ratings.csv", index=False)

    print(f"\n  Saved:")
    print(f"    fact_team_rating_snapshot.csv : {len(snapshots):,} rows")
    print(f"    fact_match_expectation.csv    : {len(expectations):,} rows")
    print(f"    elo_final_ratings.csv         : {len(final_ratings):,} teams")

    validate_elo(expectations, final_ratings)

    print("\n" + "=" * 60)
    print("  Phase 5 complete ✓")
    print("=" * 60)

    return snapshots, expectations, final_ratings


if __name__ == "__main__":
    main()
