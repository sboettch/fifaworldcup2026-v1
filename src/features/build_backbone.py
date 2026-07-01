"""
Build the complete data backbone:
- dim_team: All teams with canonical IDs, confederations, historical flags
- map_team_names: Cross-source name mapping
- fact_match: One row per match (all 49K+ international matches)
- fact_team_match: Two rows per match (one per team)

This is the central processing script per the strategy doc's Phase 1.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR, PROCESSED_DIR
from src.utils.team_names import (
    normalize_team_name, get_confederation, is_historical_team,
    get_modern_successor, TEAM_NAME_MAP, CONFEDERATION_MAP,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DIM_TEAM
# ═══════════════════════════════════════════════════════════════════════════════

def build_dim_team() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dim_team and map_team_names from all sources."""
    print("\n" + "=" * 60)
    print("  Building dim_team")
    print("=" * 60)

    # ── Collect all unique team names per source ──────────────────────────
    source_names = defaultdict(set)

    # martj42 international matches
    m = pd.read_csv(RAW_DIR / "international_matches.csv")
    source_names["martj42"] = set(m["home_team"].dropna().unique()) | set(m["away_team"].dropna().unique())

    # Fjelstul squads
    sq = pd.read_csv(RAW_DIR / "fjelstul_squads.csv")
    source_names["fjelstul"] |= set(sq["team_name"].dropna().unique())

    # Fjelstul matches
    fm = pd.read_csv(RAW_DIR / "fjelstul_matches.csv")
    source_names["fjelstul"] |= set(fm["home_team_name"].dropna().unique())
    source_names["fjelstul"] |= set(fm["away_team_name"].dropna().unique())

    # Fjelstul qualified teams
    qt = pd.read_csv(RAW_DIR / "fjelstul_qualified_teams.csv")
    source_names["fjelstul"] |= set(qt["team_name"].dropna().unique())

    # ── Normalize all names → collect canonical set ──────────────────────
    canonical_names = set()
    name_mapping_rows = []

    for source, names in source_names.items():
        for raw_name in names:
            canonical = normalize_team_name(raw_name)
            canonical_names.add(canonical)
            name_mapping_rows.append({
                "source_name": raw_name,
                "source": source,
                "canonical_team_name": canonical,
            })

    # ── Also pull Fjelstul team codes where available ────────────────────
    fjelstul_codes = {}
    for _, row in qt.iterrows():
        canonical = normalize_team_name(row["team_name"])
        if pd.notna(row.get("team_code")):
            fjelstul_codes[canonical] = row["team_code"]
    for _, row in sq.iterrows():
        canonical = normalize_team_name(row["team_name"])
        if pd.notna(row.get("team_code")):
            fjelstul_codes[canonical] = row["team_code"]

    # ── Build dim_team ───────────────────────────────────────────────────
    dim_rows = []
    for i, name in enumerate(sorted(canonical_names), start=1):
        dim_rows.append({
            "team_id": i,
            "team_name": name,
            "team_code": fjelstul_codes.get(name, ""),
            "confederation": get_confederation(name),
            "is_historical": is_historical_team(name),
            "modern_successor": get_modern_successor(name) or "",
        })

    dim_team = pd.DataFrame(dim_rows)

    # ── Build map_team_names ─────────────────────────────────────────────
    map_names = pd.DataFrame(name_mapping_rows).drop_duplicates()
    # Add team_id by joining to dim_team
    team_id_lookup = dict(zip(dim_team["team_name"], dim_team["team_id"]))
    map_names["team_id"] = map_names["canonical_team_name"].map(team_id_lookup)

    # ── QA ────────────────────────────────────────────────────────────────
    unknown_confed = dim_team[dim_team["confederation"] == "Unknown"]["team_name"].tolist()
    historical = dim_team[dim_team["is_historical"]]["team_name"].tolist()

    print(f"  Total canonical teams: {len(dim_team)}")
    print(f"  By confederation:")
    for c, cnt in dim_team["confederation"].value_counts().items():
        print(f"    {c}: {cnt}")
    print(f"  Historical teams: {len(historical)} → {historical}")
    if unknown_confed:
        print(f"  ⚠ Unknown confederation ({len(unknown_confed)}): {unknown_confed[:20]}...")
    print(f"  Name mappings: {len(map_names)} rows")

    return dim_team, map_names


# ═══════════════════════════════════════════════════════════════════════════════
#  FACT_MATCH
# ═══════════════════════════════════════════════════════════════════════════════

def build_fact_match(dim_team: pd.DataFrame) -> pd.DataFrame:
    """Build fact_match from all international matches, enriched with Fjelstul WC detail."""
    print("\n" + "=" * 60)
    print("  Building fact_match")
    print("=" * 60)

    team_id_lookup = dict(zip(dim_team["team_name"], dim_team["team_id"]))

    # ── Load primary match data (martj42 — all international matches) ────
    matches = pd.read_csv(RAW_DIR / "international_matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])

    # Normalize team names
    matches["home_team_norm"] = matches["home_team"].apply(normalize_team_name)
    matches["away_team_norm"] = matches["away_team"].apply(normalize_team_name)

    # ── Load Fjelstul WC matches for enrichment ──────────────────────────
    fjelstul = pd.read_csv(RAW_DIR / "fjelstul_matches.csv")
    fjelstul["match_date"] = pd.to_datetime(fjelstul["match_date"])
    fjelstul["home_norm"] = fjelstul["home_team_name"].apply(normalize_team_name)
    fjelstul["away_norm"] = fjelstul["away_team_name"].apply(normalize_team_name)

    # ── Load dim_stadium for stadium_id lookup ───────────────────────────
    dim_stadium = pd.read_csv(PROCESSED_DIR / "dim_stadium.csv")
    stadium_lookup = dict(zip(dim_stadium["source_stadium_id"], dim_stadium["stadium_id"]))

    # ── Load dim_tournament for tournament_id lookup ──────────────────────
    dim_tournament = pd.read_csv(PROCESSED_DIR / "dim_tournament.csv")
    # Map WC year → tournament_id
    wc_year_to_id = dict(zip(dim_tournament["year"], dim_tournament["tournament_id"]))
    # Map WC year → host_country
    wc_year_to_host = dict(zip(dim_tournament["year"], dim_tournament["host_country"]))

    # ── Create enrichment lookup from Fjelstul ───────────────────────────
    # Key: (date, home_team, away_team) → enrichment fields
    fjelstul_enrichment = {}
    for _, row in fjelstul.iterrows():
        key = (row["match_date"], row["home_norm"], row["away_norm"])
        fjelstul_enrichment[key] = {
            "stage": row.get("stage_name", ""),
            "group_name": row.get("group_name", ""),
            "extra_time": bool(row.get("extra_time", False)),
            "penalty_shootout": bool(row.get("penalty_shootout", False)),
            "home_score_penalties": row.get("home_team_score_penalties"),
            "away_score_penalties": row.get("away_team_score_penalties"),
            "stadium_id_source": row.get("stadium_id", ""),
            "stadium_name": row.get("stadium_name", ""),
            "fjelstul_city": row.get("city_name", ""),
        }

    # ── Build fact_match rows ────────────────────────────────────────────
    fact_rows = []
    enrichment_hits = 0

    for idx, row in matches.iterrows():
        home = row["home_team_norm"]
        away = row["away_team_norm"]
        date = row["date"]
        is_wc = row["tournament"] == "FIFA World Cup"

        # Base record
        record = {
            "match_id": idx + 1,
            "date": date.strftime("%Y-%m-%d"),
            "tournament": row["tournament"],
            "tournament_id": None,
            "stage": None,
            "home_team": home,
            "home_team_id": team_id_lookup.get(home),
            "away_team": away,
            "away_team_id": team_id_lookup.get(away),
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "result": None,
            "venue_city": row.get("city", ""),
            "venue_country": row.get("country", ""),
            "neutral": bool(row.get("neutral", False)),
            "is_world_cup": is_wc,
            "extra_time": False,
            "penalty_shootout": False,
            "home_score_penalties": None,
            "away_score_penalties": None,
            "stadium_id": None,
        }

        # Determine result
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            hs, aws = int(row["home_score"]), int(row["away_score"])
            if hs > aws:
                record["result"] = "H"
            elif hs < aws:
                record["result"] = "A"
            else:
                record["result"] = "D"

        # WC enrichment
        if is_wc:
            year = date.year
            record["tournament_id"] = wc_year_to_id.get(year)

            # Try Fjelstul enrichment
            key = (date, home, away)
            enrich = fjelstul_enrichment.get(key)
            if enrich:
                enrichment_hits += 1
                record["stage"] = enrich["stage"]
                record["extra_time"] = enrich["extra_time"]
                record["penalty_shootout"] = enrich["penalty_shootout"]
                record["home_score_penalties"] = enrich["home_score_penalties"]
                record["away_score_penalties"] = enrich["away_score_penalties"]
                sid = enrich.get("stadium_id_source", "")
                record["stadium_id"] = stadium_lookup.get(sid)

        fact_rows.append(record)

    fact_match = pd.DataFrame(fact_rows)

    # ── QA ────────────────────────────────────────────────────────────────
    wc_matches = fact_match[fact_match["is_world_cup"]]
    scored = fact_match[fact_match["result"].notna()]

    print(f"  Total matches: {len(fact_match):,}")
    print(f"  Date range: {fact_match['date'].min()} → {fact_match['date'].max()}")
    print(f"  Scored matches: {len(scored):,}")
    print(f"  Unscored (future): {len(fact_match) - len(scored):,}")
    print(f"  World Cup matches: {len(wc_matches):,}")
    print(f"  Fjelstul enrichment hits: {enrichment_hits}/{len(wc_matches[wc_matches['date'] <= '2022-12-31'])} WC matches (pre-2026)")
    print(f"  Unique teams: {pd.concat([fact_match['home_team'], fact_match['away_team']]).nunique()}")
    print(f"  Result distribution: {scored['result'].value_counts().to_dict()}")
    print(f"  Tournaments: {fact_match['tournament'].nunique()}")

    return fact_match


# ═══════════════════════════════════════════════════════════════════════════════
#  FACT_TEAM_MATCH
# ═══════════════════════════════════════════════════════════════════════════════

def build_fact_team_match(fact_match: pd.DataFrame, dim_team: pd.DataFrame) -> pd.DataFrame:
    """Build fact_team_match: two rows per match, one per team."""
    print("\n" + "=" * 60)
    print("  Building fact_team_match")
    print("=" * 60)

    # Load tournament hosts for is_host flag
    dim_tournament = pd.read_csv(PROCESSED_DIR / "dim_tournament.csv")
    wc_year_to_host = {}
    for _, row in dim_tournament.iterrows():
        hosts = str(row["host_country"]).split(" / ")
        wc_year_to_host[row["year"]] = [h.strip() for h in hosts]

    rows = []
    tm_id = 0

    for _, match in fact_match.iterrows():
        hs = match["home_score"]
        aws = match["away_score"]
        has_score = pd.notna(hs) and pd.notna(aws)

        # Determine host countries for this match
        year = pd.to_datetime(match["date"]).year
        host_countries = wc_year_to_host.get(year, []) if match["is_world_cup"] else []

        # Home team row
        tm_id += 1
        home_result = None
        if has_score:
            hs_int, aws_int = int(hs), int(aws)
            if hs_int > aws_int:
                home_result = "W"
            elif hs_int < aws_int:
                home_result = "L"
            else:
                home_result = "D"

        rows.append({
            "team_match_id": tm_id,
            "match_id": match["match_id"],
            "team": match["home_team"],
            "team_id": match["home_team_id"],
            "opponent": match["away_team"],
            "opponent_id": match["away_team_id"],
            "goals_for": hs if has_score else None,
            "goals_against": aws if has_score else None,
            "goal_difference": (hs - aws) if has_score else None,
            "result": home_result,
            "is_home": True,
            "is_host": match["home_team"] in host_countries,
            "tournament": match["tournament"],
            "is_world_cup": match["is_world_cup"],
            "stage": match["stage"],
            "date": match["date"],
        })

        # Away team row
        tm_id += 1
        away_result = None
        if has_score:
            if aws_int > hs_int:
                away_result = "W"
            elif aws_int < hs_int:
                away_result = "L"
            else:
                away_result = "D"

        rows.append({
            "team_match_id": tm_id,
            "match_id": match["match_id"],
            "team": match["away_team"],
            "team_id": match["away_team_id"],
            "opponent": match["home_team"],
            "opponent_id": match["home_team_id"],
            "goals_for": aws if has_score else None,
            "goals_against": hs if has_score else None,
            "goal_difference": (aws - hs) if has_score else None,
            "result": away_result,
            "is_home": False,
            "is_host": match["away_team"] in host_countries,
            "tournament": match["tournament"],
            "is_world_cup": match["is_world_cup"],
            "stage": match["stage"],
            "date": match["date"],
        })

    fact_tm = pd.DataFrame(rows)

    # ── QA checks ────────────────────────────────────────────────────────
    print(f"  Total team-match rows: {len(fact_tm):,}")
    print(f"  Expected (2 × {len(fact_match):,} matches): {2 * len(fact_match):,}")
    assert len(fact_tm) == 2 * len(fact_match), "Row count mismatch!"
    print(f"  ✓ Row count matches (2 per match)")

    # Check score reconciliation
    scored = fact_tm[fact_tm["goals_for"].notna()]
    merged = scored.merge(
        scored[["match_id", "team", "goals_for", "goals_against"]],
        left_on=["match_id", "opponent"],
        right_on=["match_id", "team"],
        suffixes=("", "_opp"),
    )
    mismatches = merged[merged["goals_for"] != merged["goals_against_opp"]]
    print(f"  Score reconciliation mismatches: {len(mismatches)}")
    if len(mismatches) > 0:
        print(f"    ⚠ {len(mismatches)} mismatches found!")
    else:
        print(f"  ✓ All scores reconcile")

    # Host flag check
    host_rows = fact_tm[fact_tm["is_host"]]
    print(f"  Matches with host team: {len(host_rows):,}")
    print(f"  Host teams: {host_rows['team'].unique().tolist()[:15]}...")

    # Result distribution
    scored_tm = fact_tm[fact_tm["result"].notna()]
    print(f"  Result distribution: {scored_tm['result'].value_counts().to_dict()}")

    wc_tm = fact_tm[fact_tm["is_world_cup"]]
    print(f"  World Cup team-match rows: {len(wc_tm):,}")

    return fact_tm


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  FIFA World Cup 2026 — Building Data Backbone")
    print("═" * 60)

    # 1. dim_team + map_team_names
    dim_team, map_names = build_dim_team()
    dim_team.to_csv(PROCESSED_DIR / "dim_team.csv", index=False)
    map_names.to_csv(PROCESSED_DIR / "map_team_names.csv", index=False)

    # 2. fact_match
    fact_match = build_fact_match(dim_team)
    fact_match.to_csv(PROCESSED_DIR / "fact_match.csv", index=False)

    # 3. fact_team_match
    fact_tm = build_fact_team_match(fact_match, dim_team)
    fact_tm.to_csv(PROCESSED_DIR / "fact_team_match.csv", index=False)

    # Final summary
    print("\n" + "═" * 60)
    print("  Backbone Build Complete")
    print("═" * 60)
    for f in ["dim_team.csv", "map_team_names.csv", "dim_tournament.csv",
              "dim_stadium.csv", "fact_match.csv", "fact_team_match.csv"]:
        path = PROCESSED_DIR / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            df = pd.read_csv(path)
            print(f"  ✓ {f}: {len(df):,} rows ({size_mb:.1f} MB)")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
