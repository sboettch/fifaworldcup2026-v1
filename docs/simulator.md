# Tournament Simulator — Technical Reference

## Overview

The Tournament Win Probability simulator forecasts how likely each remaining team is to **win the tournament** and **make the Final**, based on the bracket structure and the project's trained Random Forest model.

It runs entirely in the browser using data from `matchups.json`, which is regenerated on every pipeline run.

---

## Architecture

```
Pipeline (Python)                       Frontend (JavaScript)
────────────────────────────────────    ────────────────────────────────────────
build_2026_fixture_status               renderSimulator() in app.js
  ↓
predict_2026_scheduled                  1. Reads state.data.hypothetical_matchups
  ↓                                     2. Builds exact QF path distribution
generate_hypothetical_matchups   →JSON  3. Monte Carlo: N=2000 samples over
  ↓                                        SF + Final using model predictions
build_2026_matchup_viz                  4. Wilson CI per team
  ↓                                     5. Renders range labels (e.g. 24–31%)
build_matchup_card_data.py              6. Solid bar = mean, faded = CI band
  ↓
matchups.json (hypothetical_matchups)
```

---

## Pipeline: `generate_hypothetical_matchups.py`

**Location:** `src/models/generate_hypothetical_matchups.py`

**When it runs:** After `predict_2026_scheduled`, before `build_2026_matchup_viz`, as part of `make matchup-card-data`.

**What it does:**

1. Reads `2026_fixture_status.csv` to find confirmed QF teams
2. Assigns them to bracket sides (QF[0,1] → side 0, QF[2,3] → side 1)
3. Generates all valid SF pairings (side 0 × side 1, both directions)
4. Builds matchup features for each pair using:
   - Latest Elo ratings (`fact_team_rating_snapshot.csv`)
   - Squad fingerprints (`fact_team_fingerprint.csv`)
   - Archetype labels (via `apply_archetypes()`)
5. Trains the same RF model used for live predictions on historical WC data
6. Generates 3-class probabilities (home win / draw / away win) for each pair
7. Writes `data/processed/hypothetical_matchups.json`

**Output format:**
```json
{
  "generated_at_utc": "2026-07-10T02:52:20Z",
  "matchup_count": 32,
  "matchups": {
    "Spain vs France": {"home": 0.521, "draw": 0.198, "away": 0.281},
    "France vs Spain": {"home": 0.281, "draw": 0.198, "away": 0.521},
    ...
  }
}
```

Keys are stored in **both directions** so the JS simulator can look up either team as home.

**Fallback:** If QF teams are not yet confirmed, the script exits cleanly with an empty dict.

---

## Frontend: Hybrid Monte Carlo

**Location:** `renderSimulator()` in `app.js`

### Phase 1 — Exact QF Enumeration

All 2^N QF outcome combinations are enumerated with their exact joint probabilities. For a 3-pending-QF bracket this is 8 paths. This is exact — no sampling noise.

- Confirmed QF winners get `prob = 1.0` (always in all paths)
- Pending QFs use real model KO probabilities (draw → pens, 50/50 split)

### Phase 2 — Monte Carlo SF + Final

```
N = 2000 samples
For each sample:
  1. Draw QF path proportional to its joint probability
  2. For each SF match: look up model prediction from hypothetical_matchups
     → draw outcome via Math.random() < P(home wins)
  3. For Final: same lookup + draw
  4. Accumulate winner + finalist tallies
```

**Matchup lookup:** `"{home} vs {away}"` key in `state.data.hypothetical_matchups`.
If a key is missing, falls back to relative-strength heuristic (`s1 / (s1 + s2)`).

### Confidence Intervals

Uses the **Wilson score interval** (more robust than normal approximation at small/large p):

```
p̂ = k/N
centre = (p̂ + z²/2N) / (1 + z²/N)
margin = z√(p̂(1-p̂)/N + z²/4N²) / (1 + z²/N)
[lo, hi] = [centre − margin, centre + margin]   where z = 1.96 (95% CI)
```

**Why Wilson over normal?** At extremes (p near 0 or 1), the normal approximation can produce intervals outside [0,1]. Wilson is bounded and accurate.

### Rendering

- **Win % column:** `"24–31%"` (lo–hi of CI, rounded to integer %)
- **Makes Final column:** same format
- **Bars:** solid layer = point estimate `p̂`, faded layer = CI upper bound `hi`
- **Status line:** shows method (`model predictions · N=2000` or `strength heuristic · N=2000`)
- **SF ✓ badge:** appears on teams who have already won their QF

---

## Feature Columns Used

The hypothetical matchup generator uses the same `FEATURE_COLS` as the main model:

| Feature | Description |
|---|---|
| `elo_home_pre`, `elo_away_pre` | Elo ratings at match date |
| `elo_gap`, `elo_gap_abs` | Elo difference (signed + absolute) |
| `win_prob_home`, `win_prob_away` | Elo-based baseline probabilities |
| `k_factor` | Elo K-factor (WC = 60) |
| `is_knockout`, `is_neutral` | Match context flags |
| `heavyweight_clash` | Both teams Elo > p75 |
| `favorite_vs_underdog` | Large Elo gap archetype |
| `host_pressure` | USA / Canada / Mexico involved |
| `generational_transition` | Squad age profile archetype |
| `club_power_mismatch` | Top-5 league share gap archetype |
| `tactical_contrast` | League diversity gap archetype |
| `knockout_volatility` | Form-based volatility archetype |

**Note on form features:** `form_GD5_gap` (RF importance rank #4) is not included in `FEATURE_COLS` for scheduled predictions. The hypothetical predictions use the same feature set as `predict_2026_scheduled`.

---

## Limitations

1. **Form not updated post-QF:** SF/Final hypotheticals use squad/Elo snapshots from before the QFs. France's predicted probability against Spain doesn't account for how France performed against Morocco.

2. **Monte Carlo variance:** At N=2000, CI margins are ~±3–4pp for mid-range teams. Increasing N narrows the band but adds compute (still runs in <50ms in browser).

3. **Symmetric stage treatment:** Hypotheticals are generated with `stage="Semifinal"` for all pairs. The model's stage features (`is_knockout=1`) are correct; specific SF vs Final distinction isn't a separate feature.

---

## Files

| File | Repo | Purpose |
|---|---|---|
| `src/models/generate_hypothetical_matchups.py` | Private | Pipeline script |
| `data/processed/hypothetical_matchups.json` | Private | Script output |
| `tools/build_matchup_card_data.py` | Private | Injects into matchups.json |
| `web/matchup-card/data/matchups.json` | Both | Contains `hypothetical_matchups` key |
| `web/matchup-card/app.js` | Both | `renderSimulator()` with Monte Carlo |
| `web/matchup-card/styles.css` | Both | CI bar visual styles |
