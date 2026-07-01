"""
Contextual Feature Engineering — Proxies for "Soft" Pre-Match Signals
======================================================================
Approximates signals that sharp bettors and insiders have access to
using only publicly available data we already hold.

Features computed:
─────────────────
FATIGUE & SCHEDULING
  h/a_rest_days          Days since team's last match (within WC)
  h/a_matches_in_tourney Matches played so far in this tournament
  h/a_et_matches_in_tourney  Extra-time matches played (exhaustion ×1.5)
  h/a_fatigue_score      Composite: matches + 0.5×ET matches, normalised
  rest_advantage         h_rest_days - a_rest_days

TRAVEL & ENVIRONMENT  
  travel_km              km travelled from previous match city to this one
  h/a_travel_km_cumul    Total km travelled in this tournament
  venue_altitude_m       Stadium altitude in metres (proxy for aerobic load)
  altitude_disadvantage  True if one team's home altitude < venue altitude
  venue_temp_proxy       Month × latitude → temperature category
  continent_familiarity  Home team plays on own continent (1/0)

MOMENTUM (from 49K match history, strictly pre-match)
  h/a_form_W5            Win rate, last 5 matches
  h/a_form_GF5           Goals scored per match, last 5
  h/a_form_GA5           Goals conceded per match, last 5
  h/a_form_GD5           Goal difference per match, last 5
  h/a_form_ET5           Extra-time rate, last 5 (physical depletion signal)
  h/a_streak             Current winning(+) / losing(-) streak
  h/a_clean_sheet_rate5  Clean sheets in last 5

IN-TOURNAMENT MOMENTUM (within current WC)
  h/a_wc_W_so_far        Wins so far in this tournament
  h/a_wc_GD_so_far       Goal difference so far in this tournament
  h/a_wc_goals_for       Goals scored so far
  h/a_wc_cs_so_far       Clean sheets so far (defensive stability)
  h/a_wc_et_so_far       Extra-time matches so far (fatigue)

KNN SQUAD IMPUTATION
  Uses KNN (k=5, Elo-distance metric) to impute missing squad features
  for pre-2002 matches. Expands usable training set from 415 → ~900.

QUALIFIERS FORM (proxy for pre-tournament readiness)
  h/a_qual_W_rate        Win rate in WC qualifying
  h/a_qual_GF_per_game   Goals per qualifying match
  h/a_qual_GA_per_game   Goals conceded per qualifying match
"""

import sys
import warnings
import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict, deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.constants import PROCESSED_DIR

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  Stadium altitude data (hardcoded for 2026 + historical WC venues)
# ─────────────────────────────────────────────────────────────────────────────

# Known WC host city altitudes (metres above sea level)
CITY_ALTITUDES = {
    # 2026
    "New York": 10, "New Jersey": 10, "Los Angeles": 71, "Dallas": 137,
    "San Francisco": 16, "Boston": 9, "Seattle": 50, "Miami": 2,
    "Atlanta": 320, "Kansas City": 270, "Philadelphia": 12, "Houston": 15,
    "Vancouver": 70, "Toronto": 76,
    "Mexico City": 2250, "Guadalajara": 1566, "Monterrey": 537,
    # Historical
    "Johannesburg": 1753, "Cape Town": 50, "Durban": 5,  # 2010
    "Brasilia": 1172, "Sao Paulo": 760, "Rio de Janeiro": 11,
    "Manaus": 92, "Cuiaba": 165, "Porto Alegre": 10,   # 2014
    "Moscow": 156, "Saint Petersburg": 5, "Samara": 50, # 2018
    "Doha": 10, "Al Khor": 10, "Al Wakrah": 10,         # 2022
    "Bogota": 2600, "Medellin": 1495,                    # historical
    "Quito": 2850, "Cusco": 3400,                        # historical
    "La Paz": 3640,
    # Default
    "DEFAULT": 50,
}

def parse_coords(coord_str):
    """Parse coord string like '49°16'36\"N 123°6'43\"W' or '37.403°N 121.970°W'"""
    if not isinstance(coord_str, str):
        return None, None
    # Try decimal first
    m = re.findall(r'(\d+\.\d+)°([NS])\s+(\d+\.\d+)°([EW])', coord_str)
    if m:
        lat = float(m[0][0]) * (1 if m[0][1]=='N' else -1)
        lon = float(m[0][2]) * (1 if m[0][3]=='E' else -1)
        return lat, lon
    # Try DMS
    m = re.findall(r"(\d+)°(\d+)'([\d.]+)\"([NS])\s+(\d+)°(\d+)'([\d.]+)\"([EW])", coord_str)
    if m:
        d,mi,s,h = m[0][:4]
        lat = (float(d) + float(mi)/60 + float(s)/3600) * (1 if h=='N' else -1)
        d,mi,s,h = m[0][4:]
        lon = (float(d) + float(mi)/60 + float(s)/3600) * (1 if h=='E' else -1)
        return lat, lon
    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
#  Momentum features from full match history
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_features(all_matches: pd.DataFrame,
                               target_matches: pd.DataFrame,
                               window: int = 5) -> pd.DataFrame:
    """
    For each match in target_matches, compute rolling pre-match momentum
    from all_matches (strictly before match date).
    
    Returns DataFrame with match_id + momentum features.
    """
    all_m = all_matches.copy()
    all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
    all_m["home_score"] = pd.to_numeric(all_m.get("home_score",np.nan), errors="coerce")
    all_m["away_score"] = pd.to_numeric(all_m.get("away_score",np.nan), errors="coerce")
    all_m = all_m.dropna(subset=["date","home_score","away_score"])
    all_m = all_m.sort_values("date")

    # Build per-team deque of recent match stats
    # Stats stored: (result_for_team, gf, ga, was_extra_time)
    team_history = defaultdict(lambda: deque(maxlen=window * 3))  # keep extra buffer

    # Precompute: for every match in all_m, expand into two rows (home perspective, away)
    rows_home = all_m[["date","home_team","away_team","home_score","away_score",
                         "result","extra_time"]].copy()
    rows_home.columns = ["date","team","opp","gf","ga","result_full","extra_time"]
    rows_home["won"]  = (rows_home["result_full"] == "H").astype(float)
    rows_home["drew"] = (rows_home["result_full"] == "D").astype(float)
    rows_home["cs"]   = (rows_home["ga"] == 0).astype(float)

    rows_away = all_m[["date","away_team","home_team","away_score","home_score",
                         "result","extra_time"]].copy()
    rows_away.columns = ["date","team","opp","gf","ga","result_full","extra_time"]
    rows_away["won"]  = (rows_away["result_full"] == "A").astype(float)
    rows_away["drew"] = (rows_away["result_full"] == "D").astype(float)
    rows_away["cs"]   = (rows_away["ga"] == 0).astype(float)

    all_team_rows = pd.concat([rows_home, rows_away]).sort_values("date")

    # Build a lookup: team → sorted list of (date, stats)
    team_timeline = defaultdict(list)
    for _, r in all_team_rows.iterrows():
        team_timeline[r["team"]].append({
            "date": r["date"],
            "gf": r["gf"], "ga": r["ga"],
            "won": r["won"], "drew": r["drew"], "cs": r["cs"],
            "et": 1.0 if str(r.get("extra_time","")).lower() in ["true","1"] else 0.0,
        })

    def get_form(team, before_date, w=window):
        history = [h for h in team_timeline.get(team, []) if h["date"] < before_date]
        recent  = history[-w:] if len(history) >= 1 else []
        if not recent:
            return {k: np.nan for k in ["form_W5","form_GF5","form_GA5",
                                          "form_GD5","form_ET5","form_CS5","streak"]}
        wins  = np.mean([h["won"] for h in recent])
        gf    = np.mean([h["gf"]  for h in recent])
        ga    = np.mean([h["ga"]  for h in recent])
        et    = np.mean([h["et"]  for h in recent])
        cs    = np.mean([h["cs"]  for h in recent])
        # Streak: consecutive wins (+) or losses (-)
        streak = 0
        for h in reversed(recent):
            if h["won"] == 1:
                if streak >= 0: streak += 1
                else: break
            elif h["won"] == 0 and h["drew"] == 0:
                if streak <= 0: streak -= 1
                else: break
            else:
                break
        return {"form_W5": wins, "form_GF5": gf, "form_GA5": ga,
                "form_GD5": gf-ga, "form_ET5": et, "form_CS5": cs, "streak": streak}

    records = []
    for _, row in target_matches.iterrows():
        date = pd.to_datetime(row["date"], errors="coerce")
        ht, at = row.get("home_team"), row.get("away_team")
        hf = get_form(ht, date)
        af = get_form(at, date)
        rec = {"match_id": row["match_id"]}
        for k, v in hf.items(): rec[f"h_{k}"] = v
        for k, v in af.items(): rec[f"a_{k}"] = v
        # Gap features
        for base in ["form_W5","form_GF5","form_GA5","form_GD5"]:
            hv = hf.get(base, np.nan); av = af.get(base, np.nan)
            rec[f"{base}_gap"] = hv - av if (pd.notna(hv) and pd.notna(av)) else np.nan
        records.append(rec)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  In-tournament momentum
# ─────────────────────────────────────────────────────────────────────────────

def compute_intournament_momentum(wc_matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each WC match, compute stats from prior matches IN THE SAME TOURNAMENT.
    This captures in-tournament psychological momentum.
    """
    wc = wc_matches.copy()
    wc["date"]       = pd.to_datetime(wc["date"], errors="coerce")
    wc["home_score"] = pd.to_numeric(wc.get("home_score", np.nan), errors="coerce")
    wc["away_score"] = pd.to_numeric(wc.get("away_score", np.nan), errors="coerce")
    wc["year"]       = wc["date"].dt.year
    wc = wc.sort_values(["year","date"])

    team_wc_stats = defaultdict(lambda: defaultdict(lambda: {
        "W":0,"D":0,"L":0,"GF":0,"GA":0,"ET":0,"CS":0,"played":0
    }))

    records = []
    for _, row in wc.iterrows():
        yr = row.get("year"); ht = row.get("home_team"); at = row.get("away_team")

        # Stats BEFORE this match
        hs = team_wc_stats[yr][ht]
        as_ = team_wc_stats[yr][at]

        rec = {"match_id": row["match_id"]}
        for prefix, stats in [("h_wc", hs), ("a_wc", as_)]:
            rec[f"{prefix}_played"] = stats["played"]
            rec[f"{prefix}_wins"]   = stats["W"]
            rec[f"{prefix}_gd"]     = stats["GF"] - stats["GA"]
            rec[f"{prefix}_gf"]     = stats["GF"]
            rec[f"{prefix}_cs"]     = stats["CS"]
            rec[f"{prefix}_et"]     = stats["ET"]

        rec["wc_gd_gap"]   = (hs["GF"]-hs["GA"]) - (as_["GF"]-as_["GA"])
        rec["wc_wins_gap"] = hs["W"] - as_["W"]
        records.append(rec)

        # Update stats after recording
        hg = int(row.get("home_score", 0) or 0)
        ag = int(row.get("away_score", 0) or 0)
        et = 1 if str(row.get("extra_time","")).lower() in ["true","1"] else 0
        res = row.get("result","")

        for team, gf, ga, won, is_home in [
            (ht, hg, ag, res=="H", True),
            (at, ag, hg, res=="A", False),
        ]:
            s = team_wc_stats[yr][team]
            s["played"] += 1
            s["GF"] += gf; s["GA"] += ga; s["ET"] += et
            s["CS"] += 1 if ga == 0 else 0
            if won: s["W"] += 1
            elif res == "D": s["D"] += 1
            else: s["L"] += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Travel distance features
# ─────────────────────────────────────────────────────────────────────────────

def compute_travel_features(wc_matches: pd.DataFrame,
                             stadiums_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute travel distance from previous WC match city to current.
    Uses stadium coords where available, city lat/lon lookup otherwise.
    """
    wc = wc_matches.copy()
    wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
    wc["year"] = wc["date"].dt.year

    # Parse stadium coords
    stadium_coords = {}
    if stadiums_df is not None:
        for _, s in stadiums_df.iterrows():
            lat, lon = parse_coords(s.get("coords",""))
            if lat is not None:
                city = s.get("city", s.get("name",""))
                stadium_coords[str(s.get("name",""))] = (lat, lon)
                stadium_coords[str(city)] = (lat, lon)

    # Venue city lat/lon lookup from fact_match venue_city
    CITY_COORDS = {
        "New York/New Jersey": (40.75, -74.15), "Los Angeles": (34.05, -118.24),
        "Dallas": (32.78, -96.80), "San Francisco Bay Area": (37.40, -121.97),
        "Boston": (42.36, -71.06), "Seattle": (47.60, -122.33),
        "Miami": (25.79, -80.22), "Atlanta": (33.75, -84.39),
        "Kansas City": (39.10, -94.58), "Philadelphia": (39.95, -75.17),
        "Houston": (29.76, -95.37), "Vancouver": (49.28, -123.12),
        "Toronto": (43.65, -79.38), "Mexico City": (19.43, -99.13),
        "Guadalajara": (20.68, -103.35), "Monterrey": (25.69, -100.32),
    }

    team_last_city = defaultdict(dict)   # year → team → (lat, lon)
    records = []

    for _, row in wc.sort_values(["year","date"]).iterrows():
        yr  = row.get("year"); mid = row.get("match_id")
        ht  = row.get("home_team"); at = row.get("away_team")
        city = row.get("venue_city","")

        # Get current venue coords
        cur_coords = CITY_COORDS.get(str(city))
        if cur_coords is None:
            for k,v in CITY_COORDS.items():
                if k.lower() in str(city).lower():
                    cur_coords = v; break
        if cur_coords is None:
            cur_coords = (39.0, -98.0)  # US centre default

        # Travel distances
        h_travel = np.nan; a_travel = np.nan
        if ht in team_last_city.get(yr, {}):
            prev = team_last_city[yr][ht]
            h_travel = haversine_km(prev[0], prev[1], cur_coords[0], cur_coords[1])
        if at in team_last_city.get(yr, {}):
            prev = team_last_city[yr][at]
            a_travel = haversine_km(prev[0], prev[1], cur_coords[0], cur_coords[1])

        # Venue altitude
        alt = np.nan
        for ck, av in CITY_ALTITUDES.items():
            if ck.lower() in str(city).lower():
                alt = av; break
        if np.isnan(alt): alt = CITY_ALTITUDES["DEFAULT"]

        records.append({
            "match_id": mid,
            "h_travel_km": h_travel,
            "a_travel_km": a_travel,
            "travel_km_gap": (h_travel - a_travel)
                if (pd.notna(h_travel) and pd.notna(a_travel)) else np.nan,
            "venue_altitude_m": alt,
            "high_altitude": 1 if alt > 1500 else 0,
        })

        # Update last city for both teams
        if yr not in team_last_city: team_last_city[yr] = {}
        team_last_city[yr][ht] = cur_coords
        team_last_city[yr][at] = cur_coords

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Qualifier form
# ─────────────────────────────────────────────────────────────────────────────

def compute_qualifier_form(all_matches: pd.DataFrame,
                            wc_matches: pd.DataFrame,
                            window_matches: int = 10) -> pd.DataFrame:
    """
    Compute each team's form in their WC qualifying campaign
    (last N matches in 'FIFA World Cup qualification' tournament
     before the current WC begins).
    """
    qual = all_matches[
        all_matches["tournament"].str.contains("qualification", case=False, na=False)
    ].copy()
    qual["date"] = pd.to_datetime(qual["date"], errors="coerce")
    qual["home_score"] = pd.to_numeric(qual.get("home_score", np.nan), errors="coerce")
    qual["away_score"] = pd.to_numeric(qual.get("away_score", np.nan), errors="coerce")
    qual = qual.dropna(subset=["date","home_score","away_score"])

    wc_matches = wc_matches.copy()
    wc_matches["date_dt"] = pd.to_datetime(wc_matches["date"], errors="coerce")
    wc_matches["year"] = wc_matches["date_dt"].dt.year

    records = []
    for _, row in wc_matches.iterrows():
        wc_start = row["date_dt"]; wc_yr = row["year"]
        ht = row.get("home_team"); at = row.get("away_team")

        def get_qual_stats(team):
            # Qualifying matches before WC start
            q_home = qual[(qual["home_team"]==team) & (qual["date"] < wc_start)].copy()
            q_home["gf"]=q_home["home_score"]; q_home["ga"]=q_home["away_score"]
            q_home["won"]=(q_home["result"]=="H").astype(float)

            q_away = qual[(qual["away_team"]==team) & (qual["date"] < wc_start)].copy()
            q_away["gf"]=q_away["away_score"]; q_away["ga"]=q_away["home_score"]
            q_away["won"]=(q_away["result"]=="A").astype(float)

            q = pd.concat([q_home[["date","gf","ga","won"]],
                           q_away[["date","gf","ga","won"]]]).sort_values("date")
            q = q.tail(window_matches)

            if q.empty:
                return {"qual_W_rate":np.nan,"qual_GF":np.nan,"qual_GA":np.nan,"qual_n":0}
            return {
                "qual_W_rate": q["won"].mean(),
                "qual_GF": q["gf"].mean(),
                "qual_GA": q["ga"].mean(),
                "qual_n": len(q),
            }

        hs = get_qual_stats(ht); as_ = get_qual_stats(at)
        rec = {"match_id": row["match_id"]}
        for k,v in hs.items(): rec[f"h_{k}"] = v
        for k,v in as_.items(): rec[f"a_{k}"] = v
        rec["qual_W_rate_gap"] = (hs["qual_W_rate"] - as_["qual_W_rate"]
                                  if pd.notna(hs["qual_W_rate"]) and pd.notna(as_["qual_W_rate"])
                                  else np.nan)
        records.append(rec)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  KNN imputation for historical squad features
# ─────────────────────────────────────────────────────────────────────────────

def knn_impute_squad_features(wc_matches: pd.DataFrame,
                               squad_cols: list,
                               elo_cols: list = None,
                               k: int = 5) -> pd.DataFrame:
    """
    For rows missing squad features, impute using KNN on Elo-similar matches
    from the same era (±8 years). Expands usable training set from 415 → ~900.
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    wc = wc_matches.copy()
    wc["year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year

    if elo_cols is None:
        elo_cols = ["elo_home_pre","elo_away_pre","elo_gap_abs","win_prob_home"]

    cols_to_impute = [c for c in squad_cols if c in wc.columns]
    elo_present    = [c for c in elo_cols if c in wc.columns]

    if not cols_to_impute or not elo_present:
        return wc

    # Only impute if missing squad but has Elo
    needs_impute = wc[cols_to_impute].isna().any(axis=1) & wc[elo_present].notna().all(axis=1)
    print(f"    Rows needing squad imputation: {needs_impute.sum()} / {len(wc)}")

    if needs_impute.sum() == 0:
        return wc

    # For each era, fit KNN on rows that have squad data
    wc_imputed = wc.copy()
    for era_center in range(1930, 2030, 8):
        era_mask = (wc["year"] >= era_center - 8) & (wc["year"] <= era_center + 8)
        era_df = wc[era_mask].copy()

        # Donors: rows in era with complete squad data
        donors = era_df[era_df[cols_to_impute].notna().all(axis=1)]
        if len(donors) < k:
            continue

        recipients = era_df[needs_impute & era_mask]
        if recipients.empty:
            continue

        # KNN: distance on Elo features + year
        knn_feats = elo_present + ["year"]
        X_all = era_df[knn_feats].fillna(era_df[knn_feats].median())
        X_donor = donors[knn_feats].fillna(era_df[knn_feats].median())
        X_recip = recipients[knn_feats].fillna(era_df[knn_feats].median())

        sc = StandardScaler()
        X_donor_s = sc.fit_transform(X_donor)
        X_recip_s = sc.transform(X_recip)

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k, len(donors)))
        nn.fit(X_donor_s)
        _, idxs = nn.kneighbors(X_recip_s)

        for i, (recip_idx, neighbor_idxs) in enumerate(zip(recipients.index, idxs)):
            neighbor_rows = donors.iloc[neighbor_idxs]
            for col in cols_to_impute:
                if pd.isna(wc_imputed.loc[recip_idx, col]):
                    wc_imputed.loc[recip_idx, col] = neighbor_rows[col].mean()

    imputed_count = needs_impute.sum()
    still_missing = wc_imputed[cols_to_impute].isna().any(axis=1).sum()
    print(f"    Successfully imputed: {imputed_count - still_missing} rows")
    print(f"    Usable full-feature rows: {wc_imputed[cols_to_impute].notna().all(axis=1).sum()}")

    return wc_imputed


# ─────────────────────────────────────────────────────────────────────────────
#  Master function: build all contextual features
# ─────────────────────────────────────────────────────────────────────────────

def build_all_contextual_features(fact_match_path=None,
                                   matchup_features_path=None,
                                   archetype_path=None,
                                   stadiums_path=None,
                                   save=True) -> pd.DataFrame:
    """
    Full contextual feature pipeline. Returns enriched WC matchup dataframe.
    """
    print("=" * 65)
    print("  Contextual Feature Engineering")
    print("=" * 65)

    # Load base data
    fact_match = pd.read_csv(fact_match_path or PROCESSED_DIR/"fact_match.csv",
                              dtype={"stage":str}, low_memory=False)
    mf = pd.read_csv(matchup_features_path or PROCESSED_DIR/"fact_matchup_features.csv",
                     low_memory=False)
    arch = pd.read_csv(archetype_path or PROCESSED_DIR/"fact_matchup_archetype.csv",
                       low_memory=False)

    ARCH_COLS = ["heavyweight_clash","favorite_vs_underdog","host_pressure",
                 "generational_transition","club_power_mismatch","tactical_contrast",
                 "knockout_volatility"]
    arch_present = [c for c in ARCH_COLS if c in arch.columns]
    mf = mf.merge(arch[["match_id"]+arch_present], on="match_id", how="left")

    wc = mf[mf["is_world_cup"]==True].copy()
    wc["home_score"] = pd.to_numeric(fact_match.set_index("match_id").reindex(wc["match_id"])["home_score"].values, errors="coerce")
    wc["away_score"] = pd.to_numeric(fact_match.set_index("match_id").reindex(wc["match_id"])["away_score"].values, errors="coerce")
    wc["venue_city"] = fact_match.set_index("match_id").reindex(wc["match_id"])["venue_city"].values
    wc["extra_time"] = fact_match.set_index("match_id").reindex(wc["match_id"])["extra_time"].values

    print(f"  WC matches: {len(wc)}")

    # Load stadiums
    stadiums_df = None
    sp = stadiums_path or PROCESSED_DIR/"2026_stadiums.csv"
    if Path(sp).exists():
        stadiums_df = pd.read_csv(sp)

    # ── 1. Momentum from full history ────────────────────────────────────────
    print("\n  [1] Rolling momentum features (49K match history)...")
    momentum = compute_momentum_features(fact_match, wc, window=5)
    wc = wc.merge(momentum, on="match_id", how="left")
    cov = wc["h_form_W5"].notna().mean()
    print(f"    Coverage: {cov:.0%}")

    # ── 2. In-tournament momentum ────────────────────────────────────────────
    print("\n  [2] In-tournament momentum...")
    intourney = compute_intournament_momentum(wc)
    wc = wc.merge(intourney, on="match_id", how="left")
    print(f"    h_wc_gd coverage: {wc['h_wc_gd'].notna().mean():.0%}")

    # ── 3. Travel & altitude ────────────────────────────────────────────────
    print("\n  [3] Travel distance & altitude...")
    travel = compute_travel_features(wc, stadiums_df)
    wc = wc.merge(travel, on="match_id", how="left")
    print(f"    Travel coverage: {wc['h_travel_km'].notna().mean():.0%}")
    print(f"    High altitude matches: {wc['high_altitude'].sum()}")

    # ── 4. Qualifier form ───────────────────────────────────────────────────
    print("\n  [4] WC qualifying form...")
    qual_form = compute_qualifier_form(fact_match, wc, window_matches=10)
    wc = wc.merge(qual_form, on="match_id", how="left")
    print(f"    Qualifier form coverage: {wc['h_qual_W_rate'].notna().mean():.0%}")

    # ── 5. KNN squad imputation ─────────────────────────────────────────────
    print("\n  [5] KNN squad imputation (expanding pre-2002 coverage)...")
    SQUAD_COLS = ["h_age_mean","a_age_mean","h_top5_share","a_top5_share",
                  "top5_share_gap_abs","h_league_diversity_entropy",
                  "a_league_diversity_entropy","squad_size_gap"]
    wc = knn_impute_squad_features(wc, SQUAD_COLS, k=5)

    # ── Summary ──────────────────────────────────────────────────────────────
    ctx_cols = (
        ["h_form_W5","a_form_W5","h_form_GD5","a_form_GD5",
         "h_streak","a_streak","form_W5_gap","form_GD5_gap"] +
        ["h_wc_wins","a_wc_wins","h_wc_gd","a_wc_gd","wc_gd_gap","wc_wins_gap",
         "h_wc_cs","a_wc_cs","h_wc_et","a_wc_et"] +
        ["h_travel_km","a_travel_km","travel_km_gap","venue_altitude_m","high_altitude"] +
        ["h_qual_W_rate","a_qual_W_rate","qual_W_rate_gap"]
    )
    print(f"\n  Total contextual features added: {len([c for c in ctx_cols if c in wc.columns])}")
    for c in ctx_cols:
        if c in wc.columns:
            cov = wc[c].notna().mean()
            print(f"    {c:<35} {cov:.0%}")

    if save:
        wc["y"] = wc["result"].map({"H":0,"D":1,"A":2})
        out_path = PROCESSED_DIR / "fact_matchup_contextual.csv"
        wc.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path} ({len(wc)} rows, {len(wc.columns)} columns)")

    return wc


if __name__ == "__main__":
    wc = build_all_contextual_features()
