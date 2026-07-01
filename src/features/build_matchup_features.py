import sys, numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.constants import PROCESSED_DIR
from src.utils.team_names import normalize_team_name


def load_inputs():
    print("  Loading inputs...")
    fm     = pd.read_csv(PROCESSED_DIR / "fact_match.csv", low_memory=False)
    expect = pd.read_csv(PROCESSED_DIR / "fact_match_expectation.csv", low_memory=False)
    fp     = pd.read_csv(PROCESSED_DIR / "fact_team_fingerprint.csv", low_memory=False)
    dim_t  = pd.read_csv(PROCESSED_DIR / "dim_team.csv", low_memory=False)
    print(f"    fact_match             : {len(fm):,}")
    print(f"    fact_match_expectation : {len(expect):,}")
    print(f"    fact_team_fingerprint  : {len(fp):,}")
    return fm, expect, fp, dim_t


def make_team_lookup(dim_team):
    lkp = dim_team[["team_id", "team_name", "confederation"]].copy()
    lkp["team_name_norm"] = lkp["team_name"].apply(normalize_team_name)
    return lkp


def attach_fingerprints(matches, fp, team_lkp):
    fp_cols = [c for c in fp.columns if c not in ["team_id", "tournament_year"]]

    matches = matches.copy()
    matches["home_norm"] = matches["home_team"].apply(normalize_team_name)
    matches["away_norm"] = matches["away_team"].apply(normalize_team_name)
    matches["match_year"] = pd.to_datetime(matches["date"], errors="coerce").dt.year.fillna(0).astype(int)

    lkp_slim = team_lkp[["team_id", "team_name_norm"]].drop_duplicates("team_name_norm")
    lkp_slim["team_id"] = lkp_slim["team_id"].astype(int)

    matches = matches.merge(
        lkp_slim.rename(columns={"team_id": "home_team_id", "team_name_norm": "home_norm"}),
        on="home_norm", how="left"
    )
    matches = matches.merge(
        lkp_slim.rename(columns={"team_id": "away_team_id", "team_name_norm": "away_norm"}),
        on="away_norm", how="left"
    )

    # Ensure fp key types are int
    fp_sorted = fp.copy()
    fp_sorted["team_id"] = fp_sorted["team_id"].astype(int)
    fp_sorted["tournament_year"] = fp_sorted["tournament_year"].astype(int)
    fp_sorted = fp_sorted.sort_values("tournament_year")

    def merge_fp_side(side_col, prefix):
        side = matches[["match_id", side_col, "match_year"]].copy()
        side = side.rename(columns={side_col: "team_id"})
        side = side.dropna(subset=["team_id"])
        side["team_id"] = side["team_id"].astype(int)
        side = side.sort_values("match_year")

        merged = pd.merge_asof(
            side, fp_sorted, left_on="match_year", right_on="tournament_year",
            by="team_id", direction="backward"
        )
        rename_map = {c: f"{prefix}_{c}" for c in fp_cols}
        merged = merged.rename(columns=rename_map)
        keep = ["match_id"] + [f"{prefix}_{c}" for c in fp_cols
                                if f"{prefix}_{c}" in merged.columns]
        return merged[keep]

    print("  Merging home fingerprints...")
    home_fp = merge_fp_side("home_team_id", "h")
    print("  Merging away fingerprints...")
    away_fp = merge_fp_side("away_team_id", "a")

    result = matches.merge(home_fp, on="match_id", how="left")
    result = result.merge(away_fp, on="match_id", how="left")
    return result


def compute_matchup_features(df):
    mf = df.copy()

    def delta(col):
        h = mf.get(f"h_{col}", pd.Series(np.nan, index=mf.index))
        a = mf.get(f"a_{col}", pd.Series(np.nan, index=mf.index))
        return (h - a).round(4)

    mf["age_mean_gap"]       = delta("age_mean")
    mf["age_mean_gap_abs"]   = delta("age_mean").abs()
    mf["caps_mean_gap"]      = delta("caps_mean")
    mf["caps_mean_gap_abs"]  = delta("caps_mean").abs()
    mf["top5_share_gap"]     = delta("top5_share")
    mf["top5_share_gap_abs"] = delta("top5_share").abs()
    mf["league_div_gap"]     = delta("league_diversity_entropy")
    mf["squad_size_gap"]     = delta("squad_size")

    mf["is_knockout"] = mf.get("stage", pd.Series("", index=mf.index)).apply(
        lambda s: 1 if any(kw in str(s).lower()
                           for kw in ["final","semi","quarter","round of","knockout"]) else 0
    )
    mf["is_neutral"] = mf.get("neutral", pd.Series(False, index=mf.index)).astype(float)
    return mf


CORE_COLS = [
    "match_id","date","match_year","tournament","stage","is_world_cup",
    "home_team","away_team","home_team_id","away_team_id",
    "neutral","is_neutral","is_knockout",
    "home_score","away_score","result","extra_time","penalty_shootout",
    "elo_home_pre","elo_away_pre","elo_gap","elo_gap_abs",
    "win_prob_home","win_prob_away","k_factor",
    "age_mean_gap","age_mean_gap_abs",
    "caps_mean_gap","caps_mean_gap_abs",
    "top5_share_gap","top5_share_gap_abs",
    "league_div_gap","squad_size_gap",
    "h_age_mean","h_age_std","h_caps_mean","h_caps_max",
    "h_top5_share","h_league_diversity_entropy","h_squad_size",
    "h_pct_gk","h_pct_df","h_pct_mf","h_pct_fw",
    "a_age_mean","a_age_std","a_caps_mean","a_caps_max",
    "a_top5_share","a_league_diversity_entropy","a_squad_size",
    "a_pct_gk","a_pct_df","a_pct_mf","a_pct_fw",
]


def main():
    print("=" * 60)
    print("  Phase 5b: Build Matchup Features")
    print("=" * 60)

    fm, expect, fp, dim_team = load_inputs()
    team_lkp = make_team_lookup(dim_team)

    meta_cols = [c for c in ["match_id","stage","extra_time","penalty_shootout",
                              "neutral","tournament","is_world_cup"] if c in fm.columns]
    expect_rich = expect.merge(
        fm[meta_cols].drop_duplicates("match_id"), on="match_id", how="left", suffixes=("","_fm")
    )

    print("\n  Attaching squad fingerprints (merge_asof)...")
    enriched = attach_fingerprints(expect_rich, fp, team_lkp)
    matchup  = compute_matchup_features(enriched)

    present = [c for c in CORE_COLS if c in matchup.columns]
    final   = matchup[present]

    wc = final[final["is_world_cup"] == True]
    fp_cov = wc["h_age_mean"].notna().mean()
    print(f"\n  QA:")
    print(f"    Total rows          : {len(final):,}")
    print(f"    WC matches          : {len(wc):,}")
    print(f"    Elo coverage        : {final['elo_gap'].notna().mean():.1%}")
    print(f"    Fingerprint WC covg : {fp_cov:.1%}")
    print(f"    Columns             : {len(final.columns)}")

    final.to_csv(PROCESSED_DIR / "fact_matchup_features.csv", index=False)
    print(f"\n  Saved -> fact_matchup_features.csv")
    print("=" * 60)
    return final


if __name__ == "__main__":
    main()
