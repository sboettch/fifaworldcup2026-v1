"""
Team name normalization utilities.

Maps variant team names across all sources to canonical names.
Handles historical entities (West Germany, Soviet Union, Yugoslavia, etc.)
without collapsing them into modern successors.
"""


# ── Team name normalization map ────────────────────────────────────────────────
# Maps variant names → canonical name used in our processed data.
# The martj42 dataset uses clean modern names already, so most mappings
# are for Fjelstul/Wikipedia/Transfermarkt variants.
TEAM_NAME_MAP = {
    # Fjelstul historical names → keep as distinct entities
    "Dutch East Indies": "Dutch East Indies",
    "Chinese Taipei": "Chinese Taipei",

    # Germany variants
    "Germany FR": "West Germany",
    "Federal Republic of Germany": "West Germany",
    "German DR": "East Germany",
    "German Democratic Republic": "East Germany",

    # Russia / Soviet Union
    "USSR": "Soviet Union",

    # Yugoslavia lineage
    "FR Yugoslavia": "Serbia and Montenegro",

    # Czech lineage
    "Czechia": "Czech Republic",

    # Congo variants
    "Congo DR": "DR Congo",
    "Congo-Kinshasa": "DR Congo",

    # Ivory Coast
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",

    # Korea variants (IMPORTANT: these are DIFFERENT teams!)
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",

    # China
    "China PR": "China",

    # Iran
    "IR Iran": "Iran",

    # USA
    "USA": "United States",
    "US": "United States",

    # Turkey
    "Türkiye": "Turkey",

    # Myanmar
    "Burma": "Myanmar",

    # Swaziland
    "Swaziland": "Eswatini",

    # East Timor
    "East Timor": "Timor-Leste",

    # Bosnia
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",

    # Congo Republic (different from DR Congo!)
    "Congo": "Congo",

    # Others
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Brunei Darussalam": "Brunei",
    "Kyrgyz Republic": "Kyrgyzstan",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
}


# ── Historical teams and their modern successors ──────────────────────────────
# These are entities that no longer field teams. We keep them as separate
# dim_team entries but record the modern successor for optional grouping.
HISTORICAL_TEAMS = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "Bohemia": "Czech Republic",
    "United Arab Republic": "Egypt",
    "Tanganyika": "Tanzania",
    "Rhodesia": "Zimbabwe",
    "North Yemen": "Yemen",
    "South Yemen": "Yemen",
}


# ── FIFA Confederations ───────────────────────────────────────────────────────
CONFEDERATION_MAP = {
    # UEFA (Europe)
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czech Republic": "UEFA", "Denmark": "UEFA",
    "England": "UEFA", "Estonia": "UEFA", "Faroe Islands": "UEFA",
    "Finland": "UEFA", "France": "UEFA", "Georgia": "UEFA", "Germany": "UEFA",
    "Gibraltar": "UEFA", "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Ireland": "UEFA", "Israel": "UEFA", "Italy": "UEFA", "Kazakhstan": "UEFA",
    "Kosovo": "UEFA", "Latvia": "UEFA", "Liechtenstein": "UEFA",
    "Lithuania": "UEFA", "Luxembourg": "UEFA", "Malta": "UEFA",
    "Moldova": "UEFA", "Monaco": "UEFA", "Montenegro": "UEFA",
    "Netherlands": "UEFA", "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA",
    # Historical UEFA
    "West Germany": "UEFA", "East Germany": "UEFA", "Soviet Union": "UEFA",
    "Yugoslavia": "UEFA", "Serbia and Montenegro": "UEFA",
    "Czechoslovakia": "UEFA", "Bohemia": "UEFA",

    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",

    # CONCACAF (North/Central America & Caribbean)
    "Antigua and Barbuda": "CONCACAF", "Bahamas": "CONCACAF",
    "Barbados": "CONCACAF", "Belize": "CONCACAF", "Bermuda": "CONCACAF",
    "Canada": "CONCACAF", "Cayman Islands": "CONCACAF", "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF", "Curaçao": "CONCACAF", "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF", "El Salvador": "CONCACAF",
    "Grenada": "CONCACAF", "Guatemala": "CONCACAF", "Guyana": "CONCACAF",
    "Haiti": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Mexico": "CONCACAF", "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Suriname": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF", "United States": "CONCACAF",
    "US Virgin Islands": "CONCACAF",

    # CAF (Africa)
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Botswana": "CAF",
    "Burkina Faso": "CAF", "Burundi": "CAF", "Cameroon": "CAF",
    "Cape Verde": "CAF", "Central African Republic": "CAF", "Chad": "CAF",
    "Comoros": "CAF", "Congo": "CAF", "DR Congo": "CAF", "Djibouti": "CAF",
    "Egypt": "CAF", "Equatorial Guinea": "CAF", "Eritrea": "CAF",
    "Eswatini": "CAF", "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF",
    "Ghana": "CAF", "Guinea": "CAF", "Guinea-Bissau": "CAF",
    "Ivory Coast": "CAF", "Kenya": "CAF", "Lesotho": "CAF",
    "Liberia": "CAF", "Libya": "CAF", "Madagascar": "CAF", "Malawi": "CAF",
    "Mali": "CAF", "Mauritania": "CAF", "Mauritius": "CAF", "Morocco": "CAF",
    "Mozambique": "CAF", "Namibia": "CAF", "Niger": "CAF", "Nigeria": "CAF",
    "Rwanda": "CAF", "São Tomé and Príncipe": "CAF", "Senegal": "CAF",
    "Seychelles": "CAF", "Sierra Leone": "CAF", "Somalia": "CAF",
    "South Africa": "CAF", "South Sudan": "CAF", "Sudan": "CAF",
    "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF", "Uganda": "CAF",
    "Zambia": "CAF", "Zimbabwe": "CAF",
    # Historical CAF
    "Zaire": "CAF",

    # AFC (Asia)
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC", "Cambodia": "AFC",
    "China": "AFC", "Chinese Taipei": "AFC", "Guam": "AFC",
    "Hong Kong": "AFC", "India": "AFC", "Indonesia": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC", "Kuwait": "AFC",
    "Kyrgyzstan": "AFC", "Laos": "AFC", "Lebanon": "AFC", "Macau": "AFC",
    "Malaysia": "AFC", "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC",
    "Nepal": "AFC", "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC",
    "Palestine": "AFC", "Philippines": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Singapore": "AFC", "South Korea": "AFC",
    "Sri Lanka": "AFC", "Syria": "AFC", "Tajikistan": "AFC",
    "Thailand": "AFC", "Timor-Leste": "AFC", "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC", "Vietnam": "AFC",
    "Yemen": "AFC",
    # Historical AFC
    "Dutch East Indies": "AFC",

    # OFC (Oceania)
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "Kiribati": "OFC", "Micronesia": "OFC", "New Caledonia": "OFC",
    "New Zealand": "OFC", "Niue": "OFC", "Papua New Guinea": "OFC",
    "Samoa": "OFC", "Solomon Islands": "OFC", "Tahiti": "OFC",
    "Tonga": "OFC", "Tuvalu": "OFC", "Vanuatu": "OFC",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to its canonical form.

    Args:
        name: Raw team name from any source.

    Returns:
        Canonical team name. Returns input unchanged if no mapping exists.
    """
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def get_confederation(team_name: str) -> str:
    """Get the FIFA confederation for a team.

    Args:
        team_name: Canonical (normalized) team name.

    Returns:
        Confederation code (UEFA, CONMEBOL, etc.) or 'Unknown'.
    """
    canonical = normalize_team_name(team_name)
    return CONFEDERATION_MAP.get(canonical, "Unknown")


def is_historical_team(team_name: str) -> bool:
    """Check if a team is a historical entity that no longer exists."""
    canonical = normalize_team_name(team_name)
    return canonical in HISTORICAL_TEAMS


def get_modern_successor(team_name: str) -> str:
    """Get the modern successor of a historical team, or None."""
    canonical = normalize_team_name(team_name)
    return HISTORICAL_TEAMS.get(canonical)
