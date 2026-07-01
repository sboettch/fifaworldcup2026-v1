"""
Data collection script for Dataset 3: International Match Results.

Primary source: Mart Jürisoo's "International Football Results from 1872 to Present"
GitHub: https://github.com/martj42/international_results
This dataset is continuously updated and contains 40,000+ international matches.

Also collects the related goalscorers and shootouts files from the same repo.
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR

# ── Source URLs (GitHub raw files — no API key needed) ─────────────────────────
BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"

FILES = {
    "results.csv": "international_matches.csv",
    "goalscorers.csv": "match_goalscorers.csv",
    "shootouts.csv": "match_shootouts.csv",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  ✓ Saved to {dest}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False


def validate_matches(filepath: Path) -> None:
    """Quick validation of the downloaded match results."""
    df = pd.read_csv(filepath, parse_dates=["date"])

    print(f"\n{'='*60}")
    print(f"  Dataset 3: International Match Results — Validation")
    print(f"{'='*60}")
    print(f"  Total matches:      {len(df):,}")
    print(f"  Date range:         {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Unique teams:       {pd.concat([df['home_team'], df['away_team']]).nunique()}")
    print(f"  Tournaments:        {df['tournament'].nunique()}")
    print(f"  Countries played in:{df['country'].nunique() if 'country' in df.columns else 'N/A'}")
    print(f"  Columns:            {list(df.columns)}")
    print()

    # World Cup specific stats
    wc = df[df["tournament"] == "FIFA World Cup"]
    print(f"  World Cup matches:  {len(wc):,}")
    print(f"  WC date range:      {wc['date'].min().date()} → {wc['date'].max().date()}")
    print(f"  WC teams:           {pd.concat([wc['home_team'], wc['away_team']]).nunique()}")

    # Check for missing values
    print(f"\n  Missing values:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"    {col}: {missing:,} ({100*missing/len(df):.1f}%)")

    print(f"{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("  Downloading Dataset 3: International Match Results")
    print("="*60 + "\n")

    success_count = 0
    for source_name, local_name in FILES.items():
        url = f"{BASE_URL}/{source_name}"
        dest = RAW_DIR / local_name
        print(f"\n  Downloading {source_name} → {local_name}")
        if download_file(url, dest):
            success_count += 1

    print(f"\n  Downloaded {success_count}/{len(FILES)} files.\n")

    # Validate the main results file
    results_path = RAW_DIR / "international_matches.csv"
    if results_path.exists():
        validate_matches(results_path)

    return success_count == len(FILES)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
