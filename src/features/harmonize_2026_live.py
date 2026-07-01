"""
Harmonize 2026 live snapshots into the backbone dimensional model.

Reads the latest raw snapshot from data/raw/2026_live/, parses:
  - openfootball cup.txt → match results with scores, scorers, venues
  - openfootball cup_stadiums.csv → 2026 venue metadata
  - Wikipedia squad HTML → 2026 squad rosters (players, positions, clubs)

Then reconciles against existing fact_match.csv and fact_team_match.csv,
updating 2026 rows with enriched data from the live sources.

Usage:
    python -m src.features.harmonize_2026_live
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR, PROCESSED_DIR
from src.utils.team_names import normalize_team_name


LIVE_DIR = RAW_DIR / "2026_live"


# ═══════════════════════════════════════════════════════════════════════════════
#  Find latest snapshot
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_snapshot() -> Path | None:
    """Find the most recent run directory under data/raw/2026_live/."""
    if not LIVE_DIR.exists():
        return None

    all_runs = []
    for date_dir in sorted(LIVE_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith("run-"):
                all_runs.append(run_dir)

    if not all_runs:
        return None

    # Return the most recent by name (they're timestamped)
    return all_runs[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse openfootball cup.txt
# ═══════════════════════════════════════════════════════════════════════════════

def parse_cup_txt(cup_path: Path) -> list[dict]:
    """Parse openfootball cup.txt into structured match records.

    The format uses markers like:
        ▪ Group A
        Thu June 11
          13:00 UTC-6     Mexico  2-0 (1-0)  South Africa        @ Mexico City
                     (scorer details...)
    """
    text = cup_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    matches = []
    current_stage = ""
    current_date_str = ""

    # Patterns
    group_pattern = re.compile(r"^▪\s*(Group\s+\w+|Round of \d+|Quarter.?finals?|Semi.?finals?|Final|Third.?place)", re.IGNORECASE)
    date_pattern = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})")
    # Match line: time  Team1  score  Team2  @ Venue
    match_pattern = re.compile(
        r"^\s+\d{1,2}:\d{2}\s+UTC[+-]?\d*\s+"       # time + timezone
        r"(.+?)\s+"                                    # home team
        r"(\d+)\s*-\s*(\d+)"                          # score (home-away)
        r"(?:\s*\((\d+)-(\d+)\))?"                    # optional half-time score
        r"\s+(.+?)"                                    # away team
        r"\s+@\s+(.+?)\s*$"                           # venue
    )
    # Scorer lines (indented, parenthesized)
    scorer_pattern = re.compile(r"^\s+\((.+)")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#") or stripped.startswith("="):
            i += 1
            continue

        # Stage header
        group_match = group_pattern.match(stripped)
        if group_match:
            current_stage = group_match.group(1).strip()
            i += 1
            continue

        # Date line
        date_match = date_pattern.match(stripped)
        if date_match:
            month_name = date_match.group(1)
            day = date_match.group(2)
            current_date_str = f"2026-{month_name}-{day}"
            try:
                parsed_date = datetime.strptime(current_date_str, "%Y-%B-%d")
                current_date_str = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
            i += 1
            continue

        # Match line
        m = match_pattern.match(line)
        if m:
            home_team = m.group(1).strip()
            home_score = int(m.group(2))
            away_score = int(m.group(3))
            ht_home = int(m.group(4)) if m.group(4) else None
            ht_away = int(m.group(5)) if m.group(5) else None
            away_team = m.group(6).strip()
            venue = m.group(7).strip()

            # Collect scorer lines
            scorers = []
            j = i + 1
            while j < len(lines):
                sl = lines[j]
                if scorer_pattern.match(sl) or (sl.startswith("    ") and sl.strip() and not match_pattern.match(sl) and not date_pattern.match(sl.strip()) and not group_pattern.match(sl.strip())):
                    scorers.append(sl.strip())
                    j += 1
                else:
                    break

            # Normalize team names
            home_norm = normalize_team_name(home_team.replace("&", "and"))
            away_norm = normalize_team_name(away_team.replace("&", "and"))

            matches.append({
                "date": current_date_str,
                "stage": current_stage,
                "home_team": home_norm,
                "away_team": away_norm,
                "home_score": home_score,
                "away_score": away_score,
                "ht_home_score": ht_home,
                "ht_away_score": ht_away,
                "venue_city": venue,
                "scorers_raw": " | ".join(scorers) if scorers else "",
                "source": "openfootball",
            })

            i = j
            continue

        i += 1

    return matches


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse openfootball stadiums CSV
# ═══════════════════════════════════════════════════════════════════════════════

def parse_stadiums_csv(path: Path) -> pd.DataFrame:
    """Parse the openfootball cup_stadiums.csv with its comment format."""
    stadiums = pd.read_csv(
        path, comment="#", skip_blank_lines=True, skipinitialspace=True,
        dtype=str,
    )
    # Clean column names
    stadiums.columns = [c.strip() for c in stadiums.columns]
    # Clean values
    for col in stadiums.columns:
        stadiums[col] = stadiums[col].str.strip()

    stadiums = stadiums.dropna(subset=["city"])
    return stadiums


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse Wikipedia squad HTML
# ═══════════════════════════════════════════════════════════════════════════════

def parse_squad_html(html_path: Path) -> list[dict]:
    """Parse 2026 FIFA World Cup squads from Wikipedia HTML.

    Each country's squad is in a table with columns like:
    No. | Pos. | Player | Date of birth (age) | Caps | Goals | Club
    """
    if not html_path.exists():
        print("    ⚠ Wikipedia squad HTML not found, skipping")
        return []

    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    players = []

    # Country headings are h3, group headings are h2
    # Each h3 has a wikitable immediately following it
    country_headings = soup.find_all("h3")
    skip_names = {"contents", "references", "notes", "see also", "external links",
                  "coaching staff", "head coach", "manager"}

    for heading in country_headings:
        # Modern Wikipedia: text is directly on the h3 tag
        # Legacy Wikipedia: text is inside span.mw-headline
        span = heading.find("span", class_="mw-headline")
        if span:
            team_name = span.get_text(strip=True)
        else:
            team_name = heading.get_text(strip=True)
        if team_name.lower() in skip_names:
            continue

        current_team = normalize_team_name(team_name)

        # Find the next wikitable after this heading
        table = heading.find_next("table", class_="wikitable")
        if not table:
            continue

        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Parse header row
        headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue

            values = [cell.get_text(strip=True) for cell in cells]

            player = {
                "team": current_team,
                "tournament_year": 2026,
                "source": "wikipedia",
            }

            for h, v in zip(headers, values):
                if "no" in h or h == "#":
                    player["shirt_number"] = v
                elif "pos" in h:
                    # Clean position (e.g., "1GK" → "GK")
                    pos_clean = re.sub(r"^\d+", "", v).strip()
                    player["position"] = pos_clean if pos_clean else v
                elif "player" in h or "name" in h:
                    player["player_name"] = v
                elif "birth" in h or "age" in h:
                    player["date_of_birth_raw"] = v
                    age_match = re.search(r"aged?\s*(\d+)", v)
                    if age_match:
                        player["age"] = int(age_match.group(1))
                    # Extract clean DOB
                    dob_match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", v)
                    if dob_match:
                        player["date_of_birth"] = dob_match.group(1)
                elif "cap" in h:
                    try:
                        player["caps"] = int(v.replace(",", ""))
                    except (ValueError, AttributeError):
                        player["caps"] = None
                elif "goal" in h:
                    try:
                        player["goals"] = int(v.replace(",", ""))
                    except (ValueError, AttributeError):
                        player["goals"] = None
                elif "club" in h:
                    player["club"] = v

            if "player_name" in player and player["player_name"]:
                players.append(player)

    return players


# ═══════════════════════════════════════════════════════════════════════════════
#  Reconcile with backbone
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_matches(live_matches: list[dict]) -> pd.DataFrame:
    """Merge live 2026 match data with existing fact_match.csv.

    Updates existing 2026 WC rows with enriched data (half-time scores,
    scorers, confirmed venues) from the live snapshot.
    """
    fact_match_path = PROCESSED_DIR / "fact_match.csv"
    if not fact_match_path.exists():
        print("    [SKIP] fact_match.csv not found (CI environment — backbone not committed).")
        print("    Live match CSVs are still written; reconciliation requires local pipeline run.")
        return pd.DataFrame()
    fact_match = pd.read_csv(fact_match_path, dtype={"stage": str})

    live_df = pd.DataFrame(live_matches)
    if live_df.empty:
        print("    No live matches parsed")
        return fact_match

    # Build lookup key for live matches
    live_df["key"] = live_df["date"] + "|" + live_df["home_team"] + "|" + live_df["away_team"]
    live_lookup = live_df.set_index("key").to_dict("index")

    # Update existing 2026 WC matches
    updated = 0
    for idx, row in fact_match.iterrows():
        if not row.get("is_world_cup", False):
            continue
        if not str(row["date"]).startswith("2026"):
            continue

        key = f"{row['date']}|{row['home_team']}|{row['away_team']}"
        live = live_lookup.get(key)
        if live:
            # Update stage if we have it
            if live.get("stage"):
                fact_match.at[idx, "stage"] = live["stage"]
            # Update scores if they were NaN
            if pd.isna(row["home_score"]) and live.get("home_score") is not None:
                fact_match.at[idx, "home_score"] = live["home_score"]
                fact_match.at[idx, "away_score"] = live["away_score"]
                hs, aws = live["home_score"], live["away_score"]
                if hs > aws:
                    fact_match.at[idx, "result"] = "H"
                elif hs < aws:
                    fact_match.at[idx, "result"] = "A"
                else:
                    fact_match.at[idx, "result"] = "D"
            updated += 1

    print(f"    Reconciled {updated} matches with live data")

    # Also update fact_team_match
    fact_tm = pd.read_csv(PROCESSED_DIR / "fact_team_match.csv", dtype={"stage": str})
    tm_updated = 0
    for idx, row in fact_tm.iterrows():
        if not row.get("is_world_cup", False):
            continue
        if not str(row["date"]).startswith("2026"):
            continue
        # Find the match
        match_id = row["match_id"]
        match_row = fact_match[fact_match["match_id"] == match_id]
        if match_row.empty:
            continue
        mr = match_row.iloc[0]
        # Update stage
        if pd.notna(mr.get("stage")):
            fact_tm.at[idx, "stage"] = mr["stage"]
        # Update scores if they were NaN
        if pd.isna(row["goals_for"]) and pd.notna(mr["home_score"]):
            if row["team"] == mr["home_team"]:
                fact_tm.at[idx, "goals_for"] = mr["home_score"]
                fact_tm.at[idx, "goals_against"] = mr["away_score"]
                fact_tm.at[idx, "goal_difference"] = mr["home_score"] - mr["away_score"]
            else:
                fact_tm.at[idx, "goals_for"] = mr["away_score"]
                fact_tm.at[idx, "goals_against"] = mr["home_score"]
                fact_tm.at[idx, "goal_difference"] = mr["away_score"] - mr["home_score"]
            gf = fact_tm.at[idx, "goals_for"]
            ga = fact_tm.at[idx, "goals_against"]
            if gf > ga:
                fact_tm.at[idx, "result"] = "W"
            elif gf < ga:
                fact_tm.at[idx, "result"] = "L"
            else:
                fact_tm.at[idx, "result"] = "D"
            tm_updated += 1

    print(f"    Updated {tm_updated} team-match rows")

    fact_tm.to_csv(PROCESSED_DIR / "fact_team_match.csv", index=False)
    return fact_match


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  Harmonizing 2026 Live Data")
    print("═" * 60)

    # Ensure output directory exists (critical in CI where it's not committed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


    # Find latest snapshot
    snapshot = find_latest_snapshot()
    if not snapshot:
        print("  ❌ No live snapshots found in data/raw/2026_live/")
        return False

    print(f"  📂 Latest snapshot: {snapshot.relative_to(PROJECT_ROOT)}")

    # Check manifest
    manifest_path = snapshot / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        downloaded = sum(1 for r in manifest if r["status"] == "downloaded")
        print(f"  📊 Sources downloaded: {downloaded}/{len(manifest)}")

    # 1. Parse openfootball match results
    cup_path = snapshot / "openfootball" / "cup.txt"
    if cup_path.exists():
        print("\n  Parsing openfootball cup.txt...")
        live_matches = parse_cup_txt(cup_path)
        print(f"    ✓ Parsed {len(live_matches)} matches")
        if live_matches:
            # Save parsed matches
            live_df = pd.DataFrame(live_matches)
            live_df.to_csv(PROCESSED_DIR / "2026_live_matches.csv", index=False)
            print(f"    Saved to data/processed/2026_live_matches.csv")

            # Show stage distribution
            print(f"    Stages: {live_df['stage'].value_counts().to_dict()}")
    else:
        live_matches = []
        print("  ⚠ No cup.txt found")

    # 2. Parse stadiums
    stadiums_path = snapshot / "openfootball" / "cup_stadiums.csv"
    if stadiums_path.exists():
        print("\n  Parsing openfootball stadiums...")
        stadiums = parse_stadiums_csv(stadiums_path)
        print(f"    ✓ Parsed {len(stadiums)} stadiums")
        stadiums.to_csv(PROCESSED_DIR / "2026_stadiums.csv", index=False)
    else:
        print("  ⚠ No stadiums CSV found")

    # 3. Parse Wikipedia squads
    squad_path = snapshot / "wikipedia" / "2026_FIFA_World_Cup_squads.html"
    if squad_path.exists():
        print("\n  Parsing Wikipedia squad HTML...")
        players = parse_squad_html(squad_path)
        print(f"    ✓ Parsed {len(players)} player entries across {len(set(p['team'] for p in players))} teams")
        if players:
            squad_df = pd.DataFrame(players)
            squad_df.to_csv(PROCESSED_DIR / "2026_squads_wikipedia.csv", index=False)
            print(f"    Saved to data/processed/2026_squads_wikipedia.csv")

            # Show team counts
            team_counts = squad_df.groupby("team").size()
            print(f"    Squad sizes: min={team_counts.min()}, max={team_counts.max()}, mean={team_counts.mean():.0f}")
    else:
        print("  ⚠ No Wikipedia squad HTML found")

    # 4. Reconcile with backbone (skipped in CI if fact_match.csv not present)
    if live_matches:
        fact_match_path = PROCESSED_DIR / "fact_match.csv"
        if fact_match_path.exists():
            print("\n  Reconciling with backbone...")
            updated_fact = reconcile_matches(live_matches)
            if not updated_fact.empty:
                updated_fact.to_csv(fact_match_path, index=False)
                print("    ✓ fact_match.csv and fact_team_match.csv updated")
        else:
            print("\n  [SKIP] Backbone reconciliation skipped — fact_match.csv not present.")
            print("  Live CSVs written; run full pipeline locally to reconcile.")

    # Summary
    print("\n" + "═" * 60)
    print("  Harmonization Complete")
    print("═" * 60)
    for f in ["2026_live_matches.csv", "2026_stadiums.csv", "2026_squads_wikipedia.csv"]:
        path = PROCESSED_DIR / f
        if path.exists():
            df = pd.read_csv(path)
            print(f"  ✓ {f}: {len(df)} rows")
    print("═" * 60 + "\n")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
