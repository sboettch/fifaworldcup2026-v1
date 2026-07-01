"""
Master data collection script — runs all collectors in sequence.

Usage:
    python src/data_collection/collect_all.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection import collect_matches
from src.data_collection import collect_worldcup_db
from src.data_collection import collect_player_stats
from src.data_collection import collect_squads
from src.data_collection import collect_2026_live


def main():
    print("\n" + "=" * 70)
    print("  FIFA World Cup 2026 — Master Data Collection")
    print("=" * 70)

    results = {}

    # Dataset 3: International match results (GitHub — no auth needed)
    print("\n\n" + "─" * 70)
    print("  [1/5] Match Results (martj42/international_results)")
    print("─" * 70)
    results["matches"] = collect_matches.main()

    # Datasets 1 & 4: World Cup squads + tournament metadata (GitHub — no auth)
    print("\n\n" + "─" * 70)
    print("  [2/5] World Cup Database (jfjelstul/worldcup)")
    print("─" * 70)
    results["worldcup_db"] = collect_worldcup_db.main()

    # Dataset 1 enrichment: Wikipedia squad pages (HTML scrape)
    print("\n\n" + "─" * 70)
    print("  [3/5] Squad Context (Wikipedia World Cup squad pages)")
    print("─" * 70)
    results["squads"] = collect_squads.main()

    # Dataset 2: Player statistics (GitHub attempt + Kaggle fallback)
    print("\n\n" + "─" * 70)
    print("  [4/5] Player Statistics (transfermarkt-datasets)")
    print("─" * 70)
    results["player_stats"] = collect_player_stats.main()

    # 2026 live overlay: current tournament snapshots
    print("\n\n" + "─" * 70)
    print("  [5/5] 2026 Live Overlay (openfootball, Wikipedia, FIFA snapshot)")
    print("─" * 70)
    results["live_2026"] = collect_2026_live.main()

    # Summary
    print("\n\n" + "=" * 70)
    print("  Collection Summary")
    print("=" * 70)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    csv_files = list(raw_dir.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    print(f"\n  Total files: {len(csv_files)}")
    print(f"  Total size:  {total_size:.1f} MB")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
