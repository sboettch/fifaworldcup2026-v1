"""
Build dim_tournament and dim_stadium from raw sources.

Merges Fjelstul tournament data with hand-curated metadata to create
a comprehensive tournament dimension table.
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR, PROCESSED_DIR, REFERENCE_DIR


def build_dim_tournament() -> pd.DataFrame:
    """Build the tournament dimension table."""
    print("\n  Building dim_tournament...")

    # Load Fjelstul tournaments (1930-2022, rich detail)
    fjelstul = pd.read_csv(RAW_DIR / "fjelstul_tournaments.csv")

    # Load hand-curated reference (1930-2026, includes host continent, format)
    curated = pd.read_csv(REFERENCE_DIR / "tournament_metadata.csv")

    # Start with curated as the base (it has 2026)
    dim = curated.copy()
    dim = dim.rename(columns={
        "year": "year",
        "host_country": "host_country",
        "host_continent": "host_continent",
        "num_teams": "num_teams",
        "format": "format",
        "winner": "winner",
        "runner_up": "runner_up",
        "third_place": "third_place",
        "matches_played": "matches_played",
    })

    # Add tournament_id
    dim["tournament_id"] = range(1, len(dim) + 1)
    dim["tournament_name"] = dim["year"].apply(lambda y: f"{y} FIFA World Cup")

    # Enrich with Fjelstul dates
    fjelstul_dates = fjelstul[["year", "start_date", "end_date"]].copy()
    dim = dim.merge(fjelstul_dates, on="year", how="left")

    # For 2026, fill in known dates
    dim.loc[dim["year"] == 2026, "start_date"] = "2026-06-11"
    dim.loc[dim["year"] == 2026, "end_date"] = "2026-07-19"

    # Reorder columns
    cols = [
        "tournament_id", "tournament_name", "year", "host_country",
        "host_continent", "num_teams", "format", "winner", "runner_up",
        "third_place", "matches_played", "start_date", "end_date"
    ]
    dim = dim[cols]

    return dim


def build_dim_stadium() -> pd.DataFrame:
    """Build stadium dimension from Fjelstul match data."""
    print("  Building dim_stadium...")

    matches = pd.read_csv(RAW_DIR / "fjelstul_matches.csv")

    # Extract unique stadiums
    stadiums = matches[["stadium_id", "stadium_name", "city_name", "country_name"]].copy()
    stadiums = stadiums.drop_duplicates(subset=["stadium_id"])
    stadiums = stadiums.rename(columns={
        "stadium_id": "source_stadium_id",
        "stadium_name": "stadium_name",
        "city_name": "city",
        "country_name": "country",
    })

    # Create sequential IDs
    stadiums = stadiums.sort_values("source_stadium_id").reset_index(drop=True)
    stadiums["stadium_id"] = range(1, len(stadiums) + 1)

    cols = ["stadium_id", "source_stadium_id", "stadium_name", "city", "country"]
    stadiums = stadiums[cols]

    return stadiums


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # dim_tournament
    dim_tournament = build_dim_tournament()
    dim_tournament.to_csv(PROCESSED_DIR / "dim_tournament.csv", index=False)
    print(f"    ✓ dim_tournament: {len(dim_tournament)} rows")
    print(f"    Years: {dim_tournament['year'].min()} → {dim_tournament['year'].max()}")
    print(f"    Columns: {list(dim_tournament.columns)}")

    # dim_stadium
    dim_stadium = build_dim_stadium()
    dim_stadium.to_csv(PROCESSED_DIR / "dim_stadium.csv", index=False)
    print(f"\n    ✓ dim_stadium: {len(dim_stadium)} rows")
    print(f"    Countries: {dim_stadium['country'].nunique()}")
    print(f"    Sample:\n{dim_stadium.head().to_string()}")

    print("\n  ✓ Dimension tables built.\n")


if __name__ == "__main__":
    main()
