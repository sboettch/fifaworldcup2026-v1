"""
Collect live/provisional 2026 FIFA World Cup source snapshots.

This collector is for Stage 1A: data aggregation. It writes immutable-ish raw
source evidence under:

    data/raw/2026_live/YYYY-MM-DD/<run_id>/

The output is intentionally raw. Harmonization/parsing should happen in separate
scripts so we can always trace processed rows back to the exact source snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR


HEADERS = {
    "User-Agent": "FIFAWorldCup2026Research/1.0 (academic data aggregation; contact: local project)"
}


@dataclass(frozen=True)
class SourceSnapshot:
    source_name: str
    source_url: str
    license_note: str
    coverage_years: str
    grain: str
    source_priority: int
    relative_path: str
    required: bool


SOURCES = [
    SourceSnapshot(
        source_name="openfootball_2026_cup_txt",
        source_url="https://raw.githubusercontent.com/openfootball/worldcup/master/2026--usa/cup.txt",
        license_note="openfootball project data; verify license before redistribution",
        coverage_years="2026",
        grain="fixture-result-event-seed",
        source_priority=2,
        relative_path="openfootball/cup.txt",
        required=True,
    ),
    SourceSnapshot(
        source_name="openfootball_2026_stadiums_csv",
        source_url="https://raw.githubusercontent.com/openfootball/worldcup/master/2026--usa/cup_stadiums.csv",
        license_note="openfootball project data; verify license before redistribution",
        coverage_years="2026",
        grain="venue",
        source_priority=2,
        relative_path="openfootball/cup_stadiums.csv",
        required=True,
    ),
    SourceSnapshot(
        source_name="wikipedia_2026_world_cup_squads_html",
        source_url="https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
        license_note="CC BY-SA; use as parser input and validate against official sources",
        coverage_years="2026",
        grain="player-team-tournament",
        source_priority=3,
        relative_path="wikipedia/2026_FIFA_World_Cup_squads.html",
        required=False,
    ),
    SourceSnapshot(
        source_name="wikipedia_2026_world_cup_html",
        source_url="https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
        license_note="CC BY-SA; context source, validate canonical facts elsewhere",
        coverage_years="2026",
        grain="tournament-context",
        source_priority=3,
        relative_path="wikipedia/2026_FIFA_World_Cup.html",
        required=False,
    ),
    SourceSnapshot(
        source_name="fifa_2026_tournament_html",
        source_url="https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026",
        license_note="official FIFA page; snapshot for verification only, respect FIFA terms",
        coverage_years="2026",
        grain="official-tournament-context",
        source_priority=1,
        relative_path="fifa/tournament.html",
        required=False,
    ),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_source(source: SourceSnapshot, run_dir: Path, retrieved_at: str) -> dict:
    output_path = run_dir / source.relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        **asdict(source),
        "retrieved_at": retrieved_at,
        "raw_path": str(output_path.relative_to(PROJECT_ROOT)),
        "status": "not_started",
        "http_status": None,
        "bytes": 0,
        "sha256": None,
        "error": None,
    }

    try:
        response = requests.get(source.source_url, headers=HEADERS, timeout=45)
        record["http_status"] = response.status_code
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        return record

    payload = response.content
    output_path.write_bytes(payload)
    record["status"] = "downloaded"
    record["bytes"] = len(payload)
    record["sha256"] = sha256_bytes(payload)
    return record


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_run_manifest(record: dict) -> None:
    path = RAW_DIR.parent / "run_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def summarize_run(run_dir: Path, records: list[dict]) -> dict:
    summary = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "downloaded": sum(row["status"] == "downloaded" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "required_failed": [
            row["source_name"]
            for row in records
            if row["required"] and row["status"] != "downloaded"
        ],
        "files": [],
    }

    stadiums_path = run_dir / "openfootball" / "cup_stadiums.csv"
    if stadiums_path.exists():
        stadiums = pd.read_csv(stadiums_path, comment="#", skip_blank_lines=True, skipinitialspace=True)
        summary["files"].append(
            {
                "name": "openfootball_2026_stadiums_csv",
                "rows": int(len(stadiums)),
                "columns": list(stadiums.columns),
            }
        )

    cup_path = run_dir / "openfootball" / "cup.txt"
    if cup_path.exists():
        lines = cup_path.read_text(encoding="utf-8", errors="replace").splitlines()
        summary["files"].append(
            {
                "name": "openfootball_2026_cup_txt",
                "rows": len(lines),
                "columns": ["raw_text_line"],
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        help="Date folder for the raw snapshot, YYYY-MM-DD. Defaults to current UTC date.",
    )
    parser.add_argument(
        "--run-id",
        help="Run folder id. Defaults to run-YYYYMMDDTHHMMSSZ.",
    )
    return parser.parse_args()


def main() -> bool:
    args = parse_args()
    now = utc_now()
    snapshot_date = args.snapshot_date or now.strftime("%Y-%m-%d")
    run_id = args.run_id or f"run-{now.strftime('%Y%m%dT%H%M%SZ')}"
    retrieved_at = now.isoformat()
    run_dir = RAW_DIR / "2026_live" / snapshot_date / run_id

    print("\n" + "=" * 72)
    print("  Collecting 2026 live/provisional source snapshots")
    print(f"  Output: {run_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 72 + "\n")

    records = []
    for source in SOURCES:
        print(f"  - {source.source_name}")
        record = download_source(source, run_dir, retrieved_at)
        records.append(record)
        if record["status"] == "downloaded":
            print(f"    saved {record['bytes']:,} bytes -> {record['raw_path']}")
        else:
            print(f"    failed: {record['error']}")

    summary = summarize_run(run_dir, records)
    write_json(run_dir / "manifest.json", records)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "source_manifest_delta.csv", records)

    append_run_manifest(
        {
            "pipeline_run_id": run_id,
            "pipeline_name": "collect_2026_live",
            "snapshot_date": snapshot_date,
            "retrieved_at": retrieved_at,
            "raw_path": str(run_dir.relative_to(PROJECT_ROOT)),
            "downloaded": summary["downloaded"],
            "failed": summary["failed"],
            "required_failed": ";".join(summary["required_failed"]),
        }
    )

    print("\n" + "=" * 72)
    print("  2026 live collection summary")
    print("=" * 72)
    print(f"  Downloaded:       {summary['downloaded']}/{len(records)}")
    print(f"  Failed:           {summary['failed']}/{len(records)}")
    print(f"  Required failed:  {summary['required_failed'] or 'none'}")
    print(f"  Manifest:         {(run_dir / 'manifest.json').relative_to(PROJECT_ROOT)}")
    print(f"  Summary:          {(run_dir / 'summary.json').relative_to(PROJECT_ROOT)}")
    print("=" * 72 + "\n")

    return not summary["required_failed"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
