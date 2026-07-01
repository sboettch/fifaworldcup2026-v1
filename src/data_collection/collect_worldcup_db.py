"""
Data collection script for Dataset 1 (Squads) + Dataset 4 (Tournament Metadata).

Primary source: The Fjelstul World Cup Database
GitHub: https://github.com/jfjelstul/worldcup
License: CC-BY-NC-SA 4.0

This is the gold-standard academic dataset for World Cup research.
27 interlinked CSV files covering 1930–2022: tournaments, squads, matches,
goals, substitutions, bookings, managers, referees, awards, and standings.
"""

import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR

# ── Source: Fjelstul World Cup Database on GitHub ──────────────────────────────
BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"

# Key files we need (out of 27 available)
FILES = {
    # Squad & player data (→ Dataset 1)
    "squads.csv": "fjelstul_squads.csv",
    "squad_players.csv": "fjelstul_squad_players.csv",

    # Match-level data (enriches Dataset 3)
    "matches.csv": "fjelstul_matches.csv",
    "goals.csv": "fjelstul_goals.csv",
    "substitutions.csv": "fjelstul_substitutions.csv",
    "bookings.csv": "fjelstul_bookings.csv",

    # Tournament metadata (→ Dataset 4)
    "tournaments.csv": "fjelstul_tournaments.csv",
    "groups.csv": "fjelstul_groups.csv",
    "group_standings.csv": "fjelstul_group_standings.csv",
    "stages.csv": "fjelstul_stages.csv",

    # Additional useful context
    "managers.csv": "fjelstul_managers.csv",
    "awards.csv": "fjelstul_awards.csv",
    "qualified_teams.csv": "fjelstul_qualified_teams.csv",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a single CSV file."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed: {e}")
        return False


def validate_data(raw_dir: Path) -> None:
    """Print summary statistics for key downloaded files."""
    print(f"\n{'='*60}")
    print(f"  Fjelstul World Cup Database — Validation")
    print(f"{'='*60}")

    key_files = {
        "Squad players": "fjelstul_squad_players.csv",
        "Matches": "fjelstul_matches.csv",
        "Goals": "fjelstul_goals.csv",
        "Tournaments": "fjelstul_tournaments.csv",
        "Qualified teams": "fjelstul_qualified_teams.csv",
    }

    for label, filename in key_files.items():
        filepath = raw_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"\n  {label} ({filename}):")
            print(f"    Rows: {len(df):,}")
            print(f"    Columns: {list(df.columns)}")
            if "year" in df.columns:
                print(f"    Year range: {df['year'].min()} → {df['year'].max()}")
            if "team_name" in df.columns:
                print(f"    Teams: {df['team_name'].nunique()}")
            elif "home_team_name" in df.columns:
                teams = pd.concat([
                    df.get("home_team_name", pd.Series()),
                    df.get("away_team_name", pd.Series())
                ]).dropna().nunique()
                print(f"    Teams: {teams}")

    print(f"\n{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("  Downloading Fjelstul World Cup Database")
    print("  Source: github.com/jfjelstul/worldcup")
    print("="*60 + "\n")

    success_count = 0
    for source_name, local_name in tqdm(FILES.items(), desc="Downloading"):
        url = f"{BASE_URL}/{source_name}"
        dest = RAW_DIR / local_name
        if download_file(url, dest):
            success_count += 1
            tqdm.write(f"  ✓ {source_name} → {local_name}")
        else:
            tqdm.write(f"  ✗ {source_name} — failed")

    print(f"\n  Downloaded {success_count}/{len(FILES)} files.\n")

    validate_data(RAW_DIR)

    return success_count == len(FILES)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
