"""
Phase 4: Squad & Player Feature Builder
========================================
Builds:
  - dim_player          : one row per canonical player (name + team + birth year)
  - bridge_squad        : one row per player-team-tournament appearance
  - fact_team_fingerprint: one row per team-tournament with aggregated squad features

Squad sources:
  - data/processed/2026_squads_wikipedia.csv  (1,290 rows, 48 teams, 2026 only)
  - data/raw/fjelstul_squads.csv              (historical WC squads, 1930-2022)

Output files (data/processed/):
  - dim_player.csv
  - bridge_squad.csv
  - fact_team_fingerprint.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR, PROCESSED_DIR
from src.utils.team_names import normalize_team_name

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TOP5_LEAGUES = {
    "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
    "english premier league", "spain primera division", "german bundesliga",
    "italian serie a", "french ligue 1",
}

LEAGUE_KEYWORDS = {
    "premier": "Premier League",
    "la liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie a": "Serie A",
    "ligue 1": "Ligue 1",
    "eredivisie": "Eredivisie",
    "primeira liga": "Primeira Liga",
    "super lig": "Super Lig",
    "mls": "MLS",
    "saudi": "Saudi Pro League",
}


def infer_league(club: str) -> str:
    """Best-effort league inference from club name."""
    if pd.isna(club):
        return "Unknown"
    c = club.lower()
    for kw, league in LEAGUE_KEYWORDS.items():
        if kw in c:
            return league
    return "Other"


def is_top5(club: str) -> bool:
    league = infer_league(club).lower()
    return any(t in league for t in ["premier", "liga", "bundesliga", "serie a", "ligue 1"])


def safe_int(x):
    try:
        v = float(x)
        return int(v) if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & NORMALISE SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def load_fjelstul_squads() -> pd.DataFrame:
    """Load Fjelstul historical WC squads (1930-2022).

    Actual Fjelstul columns:
        key_id, tournament_id, tournament_name, team_id, team_name,
        team_code, player_id, family_name, given_name,
        shirt_number, position_name, position_code
    """
    path = RAW_DIR / "fjelstul_squads.csv"
    if not path.exists():
        print(f"  [WARN] {path} not found - skipping historical squads")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    print(f"  Fjelstul squads raw: {len(df):,} rows")

    out = pd.DataFrame()
    out["team_raw"] = df["team_name"]

    # Combine given + family name
    out["player_name"] = (
        df["given_name"].fillna("") + " " + df["family_name"].fillna("")
    ).str.strip()

    # Extract year from e.g. "1930 FIFA Men's World Cup"
    out["tournament_year"] = (
        df["tournament_name"].str.extract(r"(\d{4})")[0].astype("Int64")
    )

    # Normalize position
    pos_map = {
        "goal keeper": "GK", "goalkeeper": "GK",
        "defender": "DF",
        "midfielder": "MF",
        "forward": "FW", "attacker": "FW",
    }
    raw_pos = df["position_name"].str.lower().str.strip()
    out["position"] = raw_pos.map(pos_map).fillna(df["position_code"].str.upper())

    out["shirt_number"] = pd.to_numeric(df["shirt_number"], errors="coerce")
    # Fjelstul has no caps, dob, or club columns
    out["date_of_birth"] = None
    out["caps"] = None
    out["club"] = None
    out["source"] = "fjelstul"

    print(f"  Fjelstul: {out['tournament_year'].nunique()} editions "
          f"({out['tournament_year'].min()}-{out['tournament_year'].max()}), "
          f"{out['team_raw'].nunique()} teams")

    return out[["team_raw", "tournament_year", "player_name", "position",
                "shirt_number", "date_of_birth", "caps", "club", "source"]]


def load_wikipedia_squads() -> pd.DataFrame:
    """Load 2026 Wikipedia squads."""
    path = PROCESSED_DIR / "2026_squads_wikipedia.csv"
    df = pd.read_csv(path, low_memory=False)
    print(f"  Wikipedia squads: {len(df):,} rows, {df['team'].nunique()} teams")

    df = df.rename(columns={
        "team": "team_raw",
        "date_of_birth": "date_of_birth",
    })
    df["source"] = "wikipedia_2026"

    for col in ["goals"]:
        if col not in df.columns:
            df[col] = None

    return df[["team_raw", "tournament_year", "player_name", "position",
               "shirt_number", "date_of_birth", "caps", "club", "source"]]


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD DIM_PLAYER
# ─────────────────────────────────────────────────────────────────────────────

def build_dim_player(combined: pd.DataFrame) -> pd.DataFrame:
    """
    One row per canonical player identity.
    Key: (player_name_normalized, team_canonical, birth_year)
    """
    print("\n  Building dim_player...")

    df = combined.copy()
    df["player_name_norm"] = df["player_name"].str.strip().str.lower().fillna("unknown")
    df["team_canonical"] = df["team_raw"].apply(normalize_team_name)

    # Parse birth year
    df["birth_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year

    # Deduplicate: one row per unique (name, team, birth_year) triple
    players = (
        df.groupby(["player_name_norm", "team_canonical", "birth_year"], dropna=False)
        .agg(
            player_name=("player_name", "first"),
            caps_max=("caps", "max"),
            club_latest=("club", "last"),
            position_primary=("position", "first"),
        )
        .reset_index()
    )

    players.insert(0, "player_id", range(1, len(players) + 1))

    print(f"  → {len(players):,} canonical players")
    return players


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD BRIDGE_SQUAD
# ─────────────────────────────────────────────────────────────────────────────

def build_bridge_squad(combined: pd.DataFrame, dim_player: pd.DataFrame,
                       dim_team: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-team-tournament appearance.
    Links player_id → team_id → tournament_year.
    """
    print("\n  Building bridge_squad...")

    df = combined.copy()
    df["player_name_norm"] = df["player_name"].str.strip().str.lower().fillna("unknown")
    df["team_canonical"] = df["team_raw"].apply(normalize_team_name)
    df["birth_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year
    df["age"] = df["tournament_year"] - df["birth_year"]
    df["is_top5_club"] = df["club"].apply(is_top5)
    df["inferred_league"] = df["club"].apply(infer_league)
    df["caps"] = pd.to_numeric(df["caps"], errors="coerce")

    # Join player_id
    key_cols = ["player_name_norm", "team_canonical", "birth_year"]
    df = df.merge(
        dim_player[["player_id"] + key_cols],
        on=key_cols,
        how="left",
    )

    # Join team_id
    team_lookup = dim_team[["team_id", "team_name"]].copy()
    team_lookup["team_canonical"] = team_lookup["team_name"].apply(normalize_team_name)
    df = df.merge(team_lookup[["team_id", "team_canonical"]], on="team_canonical", how="left")

    bridge = df[[
        "player_id", "team_id", "tournament_year",
        "shirt_number", "position", "age", "caps",
        "club", "inferred_league", "is_top5_club", "source",
    ]].copy()

    bridge["shirt_number"] = pd.to_numeric(bridge["shirt_number"], errors="coerce")
    bridge["caps"] = pd.to_numeric(bridge["caps"], errors="coerce")
    bridge = bridge.drop_duplicates(subset=["player_id", "team_id", "tournament_year"])

    print(f"  → {len(bridge):,} squad entries across "
          f"{bridge['tournament_year'].nunique()} tournaments, "
          f"{bridge['team_id'].nunique()} teams")
    return bridge


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD FACT_TEAM_FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

def build_team_fingerprint(bridge: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (team_id, tournament_year) — the 'team fingerprint'.
    Aggregates squad-level features into team-level summary statistics.
    """
    print("\n  Building fact_team_fingerprint...")

    g = bridge.groupby(["team_id", "tournament_year"])

    def entropy(series):
        """Shannon entropy of a categorical series (for league diversity)."""
        counts = series.value_counts(normalize=True)
        return float(-(counts * np.log2(counts + 1e-10)).sum())

    fp = g.agg(
        squad_size=("player_id", "count"),
        # Age features
        age_mean=("age", "mean"),
        age_median=("age", "median"),
        age_std=("age", "std"),
        age_min=("age", "min"),
        age_max=("age", "max"),
        # Caps (international experience)
        caps_mean=("caps", "mean"),
        caps_median=("caps", "median"),
        caps_max=("caps", "max"),
        caps_sum=("caps", "sum"),
        # Club quality
        top5_share=("is_top5_club", "mean"),
    ).reset_index()

    # League diversity (entropy) — compute separately
    league_entropy = (
        bridge.groupby(["team_id", "tournament_year"])["inferred_league"]
        .apply(entropy)
        .reset_index()
        .rename(columns={"inferred_league": "league_diversity_entropy"})
    )
    fp = fp.merge(league_entropy, on=["team_id", "tournament_year"], how="left")

    # Position counts
    pos_counts = (
        bridge.groupby(["team_id", "tournament_year", "position"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Normalise known position labels
    pos_map = {"GK": "n_gk", "DF": "n_df", "MF": "n_mf", "FW": "n_fw",
               "Goalkeeper": "n_gk", "Defender": "n_df",
               "Midfielder": "n_mf", "Forward": "n_fw",
               "G": "n_gk", "D": "n_df", "M": "n_mf", "F": "n_fw"}
    pos_counts = pos_counts.rename(columns={k: v for k, v in pos_map.items()
                                            if k in pos_counts.columns})
    for col in ["n_gk", "n_df", "n_mf", "n_fw"]:
        if col not in pos_counts.columns:
            pos_counts[col] = 0

    fp = fp.merge(
        pos_counts[["team_id", "tournament_year", "n_gk", "n_df", "n_mf", "n_fw"]],
        on=["team_id", "tournament_year"],
        how="left",
    )

    # Pct shares
    for col in ["n_gk", "n_df", "n_mf", "n_fw"]:
        fp[col.replace("n_", "pct_")] = fp[col] / fp["squad_size"].replace(0, np.nan)

    # Round floats
    float_cols = [c for c in fp.columns if fp[c].dtype == float]
    fp[float_cols] = fp[float_cols].round(4)

    print(f"  → {len(fp):,} team-tournament fingerprints")
    print(f"     Coverage: {fp['tournament_year'].min()}–{fp['tournament_year'].max()}")
    print(f"     Columns: {len(fp.columns)}")
    return fp


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 4: Build Players & Squad Features")
    print("=" * 60)

    dim_team = pd.read_csv(PROCESSED_DIR / "dim_team.csv")

    # Load sources
    fjelstul = load_fjelstul_squads()
    wiki = load_wikipedia_squads()

    if fjelstul.empty:
        combined = wiki.copy()
    else:
        combined = pd.concat([fjelstul, wiki], ignore_index=True)

    # Normalise tournament_year
    combined["tournament_year"] = pd.to_numeric(
        combined["tournament_year"], errors="coerce"
    ).astype("Int64")

    print(f"\n  Combined squads: {len(combined):,} rows | "
          f"{combined['tournament_year'].nunique()} editions | "
          f"{combined['team_raw'].nunique()} raw team names")

    # Build outputs
    dim_player = build_dim_player(combined)
    bridge = build_bridge_squad(combined, dim_player, dim_team)
    fingerprint = build_team_fingerprint(bridge)

    # ── Save ──────────────────────────────────────────────────────────────
    dim_player.to_csv(PROCESSED_DIR / "dim_player.csv", index=False)
    bridge.to_csv(PROCESSED_DIR / "bridge_squad.csv", index=False)
    fingerprint.to_csv(PROCESSED_DIR / "fact_team_fingerprint.csv", index=False)

    print("\n" + "=" * 60)
    print("  QA Summary")
    print("=" * 60)
    matched = bridge["player_id"].notna().mean()
    team_matched = bridge["team_id"].notna().mean()
    print(f"  Player ID match rate  : {matched:.1%}")
    print(f"  Team ID match rate    : {team_matched:.1%}")
    print(f"  dim_player            : {len(dim_player):,} rows → dim_player.csv")
    print(f"  bridge_squad          : {len(bridge):,} rows → bridge_squad.csv")
    print(f"  fact_team_fingerprint : {len(fingerprint):,} rows → fact_team_fingerprint.csv")
    print("=" * 60)

    return dim_player, bridge, fingerprint


if __name__ == "__main__":
    main()
