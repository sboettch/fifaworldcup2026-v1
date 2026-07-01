"""
Data collection script for Dataset 2: Individual Player Statistics.

Primary source: Transfermarkt Datasets by David Cariboo
GitHub: https://github.com/dcaribou/transfermarkt-datasets
Kaggle: https://www.kaggle.com/datasets/davidcariboo/player-scores

Contains 90,000+ players, 80,000+ games, 1.9M+ appearance records.
Covers both club AND national team appearances.
Updated weekly.

We download the key relational CSVs:
- players.csv: player profiles, positions, market values
- appearances.csv: per-game stats (goals, assists, minutes, cards)
- games.csv: match-level context
- player_valuations.csv: historical market values
"""

import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR

# ── Source: transfermarkt-datasets on GitHub ────────────────────────────────────
# The repo provides pre-built CSV snapshots via GitHub releases
BASE_URL = "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data"

# These are the prepared/cleaned files from the repo
# Note: The actual data files are large; the repo uses DVC for storage.
# We'll try the GitHub raw files first, then fall back to Kaggle instructions.
PREPARED_BASE = "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/refs/heads/master/prep/data"

# Alternative: Direct download from a known Kaggle mirror snapshot
# Users can also run: kaggle datasets download -d davidcariboo/player-scores
KAGGLE_DATASET = "davidcariboo/player-scores"

FILES_TO_DOWNLOAD = {
    "players.csv": "transfermarkt_players.csv",
    "appearances.csv": "transfermarkt_appearances.csv",
    "games.csv": "transfermarkt_games.csv",
    "player_valuations.csv": "transfermarkt_valuations.csv",
    "clubs.csv": "transfermarkt_clubs.csv",
    "competitions.csv": "transfermarkt_competitions.csv",
}


def try_download(url: str, dest: Path) -> bool:
    """Attempt to download a file, return success boolean."""
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False


def download_from_github() -> int:
    """Try downloading from GitHub (may not work for large DVC-tracked files)."""
    success = 0
    for source_name, local_name in FILES_TO_DOWNLOAD.items():
        # Try multiple potential paths
        urls_to_try = [
            f"{BASE_URL}/{source_name}",
            f"{PREPARED_BASE}/{source_name}",
        ]
        dest = RAW_DIR / local_name

        for url in urls_to_try:
            if try_download(url, dest):
                tqdm.write(f"  ✓ {source_name} → {local_name}")
                success += 1
                break
        else:
            tqdm.write(f"  ✗ {source_name} — not available via GitHub raw")

    return success


def print_kaggle_instructions():
    """Print instructions for manual Kaggle download."""
    print(f"""
{'='*60}
  Transfermarkt data requires Kaggle download
{'='*60}

  The Transfermarkt dataset files are large and tracked via DVC,
  so they can't be downloaded directly from GitHub raw URLs.

  To download via Kaggle CLI:

    1. Install Kaggle CLI (if not already):
       pip install kaggle

    2. Set up API credentials:
       - Go to https://www.kaggle.com/settings
       - Click "Create New Token" under API
       - Save kaggle.json to ~/.kaggle/kaggle.json

    3. Download the dataset:
       kaggle datasets download -d {KAGGLE_DATASET} \\
         -p {RAW_DIR} --unzip

  Or download manually from:
    https://www.kaggle.com/datasets/{KAGGLE_DATASET}

  After downloading, rename files:
""")
    for source, local in FILES_TO_DOWNLOAD.items():
        print(f"    {source} → {local}")
    print(f"\n{'='*60}\n")


def validate_data(raw_dir: Path) -> None:
    """Validate whatever Transfermarkt data we have."""
    print(f"\n{'='*60}")
    print(f"  Transfermarkt Player Data — Validation")
    print(f"{'='*60}")

    found = False
    for local_name in FILES_TO_DOWNLOAD.values():
        filepath = raw_dir / local_name
        if filepath.exists():
            found = True
            df = pd.read_csv(filepath, nrows=5)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            full_df = pd.read_csv(filepath)
            print(f"\n  {local_name}:")
            print(f"    Size: {size_mb:.1f} MB")
            print(f"    Rows: {len(full_df):,}")
            print(f"    Columns: {list(full_df.columns)}")

    if not found:
        print("\n  No Transfermarkt files found yet.")
        print("  Run this script or follow the Kaggle download instructions.")

    print(f"\n{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("  Downloading Dataset 2: Player Statistics (Transfermarkt)")
    print("="*60 + "\n")

    # Try GitHub first
    print("  Attempting GitHub download...")
    success = download_from_github()

    if success < len(FILES_TO_DOWNLOAD):
        print(f"\n  Only {success}/{len(FILES_TO_DOWNLOAD)} files available via GitHub.")
        print_kaggle_instructions()

    # Validate whatever we got
    validate_data(RAW_DIR)

    return success > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
