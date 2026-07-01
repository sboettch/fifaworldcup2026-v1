"""
Data collection script for Dataset 1: World Cup Squad Rosters.

Scrapes Wikipedia's well-structured squad pages for each World Cup edition.
Each World Cup has a dedicated page like:
  https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads

These pages contain standardised HTML tables with player name, position,
date of birth, caps, goals, and club team.
"""

import sys
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RAW_DIR, WORLD_CUP_YEARS

# ── Wikipedia squad page URL patterns ─────────────────────────────────────────
# URLs vary slightly by year, but follow a common pattern
SQUAD_URLS = {
    1930: "https://en.wikipedia.org/wiki/1930_FIFA_World_Cup_squads",
    1934: "https://en.wikipedia.org/wiki/1934_FIFA_World_Cup_squads",
    1938: "https://en.wikipedia.org/wiki/1938_FIFA_World_Cup_squads",
    1950: "https://en.wikipedia.org/wiki/1950_FIFA_World_Cup_squads",
    1954: "https://en.wikipedia.org/wiki/1954_FIFA_World_Cup_squads",
    1958: "https://en.wikipedia.org/wiki/1958_FIFA_World_Cup_squads",
    1962: "https://en.wikipedia.org/wiki/1962_FIFA_World_Cup_squads",
    1966: "https://en.wikipedia.org/wiki/1966_FIFA_World_Cup_squads",
    1970: "https://en.wikipedia.org/wiki/1970_FIFA_World_Cup_squads",
    1974: "https://en.wikipedia.org/wiki/1974_FIFA_World_Cup_squads",
    1978: "https://en.wikipedia.org/wiki/1978_FIFA_World_Cup_squads",
    1982: "https://en.wikipedia.org/wiki/1982_FIFA_World_Cup_squads",
    1986: "https://en.wikipedia.org/wiki/1986_FIFA_World_Cup_squads",
    1990: "https://en.wikipedia.org/wiki/1990_FIFA_World_Cup_squads",
    1994: "https://en.wikipedia.org/wiki/1994_FIFA_World_Cup_squads",
    1998: "https://en.wikipedia.org/wiki/1998_FIFA_World_Cup_squads",
    2002: "https://en.wikipedia.org/wiki/2002_FIFA_World_Cup_squads",
    2006: "https://en.wikipedia.org/wiki/2006_FIFA_World_Cup_squads",
    2010: "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup_squads",
    2014: "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup_squads",
    2018: "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads",
    2022: "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads",
    2026: "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
}

HEADERS = {
    "User-Agent": "FIFAWorldCup2026Research/1.0 (Academic project; Python/requests)"
}


def extract_country_name(heading) -> str:
    """Extract country name from a Wikipedia section heading element."""
    # Remove edit links and footnotes, get clean text
    for tag in heading.find_all(["sup", "span"]):
        if tag.get("class") and "mw-editsection" in tag.get("class", []):
            tag.decompose()
    text = heading.get_text(strip=True)
    # Remove footnote markers like [1]
    import re
    text = re.sub(r"\[.*?\]", "", text).strip()
    return text


def scrape_year(year: int, url: str) -> pd.DataFrame:
    """Scrape squad data for a single World Cup year from Wikipedia."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to fetch {year}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, "lxml")
    all_players = []

    # Find all h3 headings (country names) — each followed by a squad table
    headings = soup.find_all(["h3", "h2"])

    for heading in headings:
        country = extract_country_name(heading)

        # Skip non-country headings
        skip_words = [
            "references", "notes", "see also", "external links",
            "contents", "group", "head coach", "manager", "edit",
            "navigation", "squad", "overview",
        ]
        if any(w in country.lower() for w in skip_words) or len(country) < 2:
            continue

        # Find the next table after this heading
        table = heading.find_next("table", class_="wikitable")
        if table is None:
            continue

        # Parse the table
        try:
            rows = table.find_all("tr")
            header_cells = rows[0].find_all(["th", "td"]) if rows else []
            headers = [cell.get_text(strip=True).lower() for cell in header_cells]

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    continue

                player_data = {
                    "year": year,
                    "country": country,
                }

                # Map columns dynamically based on header content
                for i, cell in enumerate(cells):
                    if i >= len(headers):
                        break
                    h = headers[i]
                    val = cell.get_text(strip=True)

                    if any(k in h for k in ["#", "no", "number"]):
                        player_data["shirt_number"] = val
                    elif any(k in h for k in ["pos", "position"]):
                        player_data["position"] = val
                    elif any(k in h for k in ["player", "name"]):
                        player_data["player_name"] = val
                    elif any(k in h for k in ["birth", "born", "date of birth"]):
                        player_data["date_of_birth"] = val
                    elif any(k in h for k in ["cap"]):
                        player_data["caps"] = val
                    elif any(k in h for k in ["goal"]):
                        player_data["goals"] = val
                    elif any(k in h for k in ["club"]):
                        player_data["club_team"] = val
                    elif any(k in h for k in ["age"]):
                        player_data["age"] = val

                if "player_name" in player_data and player_data["player_name"]:
                    all_players.append(player_data)

        except Exception as e:
            print(f"    Warning: Error parsing table for {country} ({year}): {e}")
            continue

    df = pd.DataFrame(all_players)
    return df


def main():
    print("\n" + "="*60)
    print("  Downloading Dataset 1: World Cup Squad Rosters")
    print("  Source: Wikipedia")
    print("="*60 + "\n")

    all_squads = []

    for year in tqdm(sorted(SQUAD_URLS.keys()), desc="Scraping squads"):
        url = SQUAD_URLS[year]
        print(f"\n  {year}: {url}")

        df = scrape_year(year, url)
        if not df.empty:
            all_squads.append(df)
            print(f"    ✓ {len(df)} players from {df['country'].nunique()} countries")
        else:
            print(f"    ✗ No data extracted")

        # Be respectful to Wikipedia
        time.sleep(1.5)

    if all_squads:
        combined = pd.concat(all_squads, ignore_index=True)
        output_path = RAW_DIR / "world_cup_squads.csv"
        combined.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"  Squad Roster Collection — Summary")
        print(f"{'='*60}")
        print(f"  Total players:    {len(combined):,}")
        print(f"  Years covered:    {sorted(combined['year'].unique())}")
        print(f"  Countries:        {combined['country'].nunique()}")
        print(f"  Columns:          {list(combined.columns)}")
        print(f"  Saved to:         {output_path}")
        print(f"{'='*60}\n")
    else:
        print("\n  ✗ No squad data was collected.\n")

    return len(all_squads) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
