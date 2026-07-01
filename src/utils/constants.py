"""
Shared constants and configuration for the FIFA World Cup 2026 project.
"""

import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# ── Dataset filenames ──────────────────────────────────────────────────────────
# Dataset 3: Match results
MATCH_RESULTS_RAW = RAW_DIR / "international_matches.csv"
MATCH_RESULTS_PROCESSED = PROCESSED_DIR / "matches_clean.csv"

# Dataset 1: Squads
SQUADS_RAW = RAW_DIR / "world_cup_squads.csv"
SQUADS_PROCESSED = PROCESSED_DIR / "squads_clean.csv"

# Dataset 2: Player stats
PLAYER_STATS_RAW = RAW_DIR / "player_stats.csv"
PLAYER_STATS_PROCESSED = PROCESSED_DIR / "player_stats_clean.csv"

# Dataset 4: Tournament metadata
TOURNAMENT_META = REFERENCE_DIR / "tournament_metadata.csv"

# ── World Cup editions ─────────────────────────────────────────────────────────
WORLD_CUP_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974,
    1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014,
    2018, 2022, 2026
]

MODERN_ERA_START = 1998  # 32-team format began

# ── FIFA confederations ────────────────────────────────────────────────────────
CONFEDERATIONS = {
    "UEFA": "Europe",
    "CONMEBOL": "South America",
    "CONCACAF": "North/Central America & Caribbean",
    "CAF": "Africa",
    "AFC": "Asia",
    "OFC": "Oceania",
}

# ── 2026 World Cup specifics ──────────────────────────────────────────────────
WC_2026_HOSTS = ["United States", "Canada", "Mexico"]
WC_2026_NUM_TEAMS = 48
WC_2026_FORMAT = "12 groups of 4, expanded knockout"
