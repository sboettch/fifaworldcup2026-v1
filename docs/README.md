# World Cup 2026 Matchup Tracker — Documentation

## What is this?

A live prediction tracker for the 2026 FIFA World Cup. It combines:

- A trained Random Forest model (RF + Elo + squad fingerprints + archetype labels)
- Live match data updated every 6 hours via GitHub Actions
- An interactive matchup explorer with historical context
- A tournament simulator with real model-based probability forecasts

---

## Live Site

**[worldcup26tracker.com](https://worldcup26tracker.com)**

---

## Variants

The site ships two variants accessible via the navigation:

| URL | What it shows |
|---|---|
| `/` or `/index-b.html` | Full tracker + **Tournament Simulator** |
| `/index-a.html` | Full tracker, no simulator |

---

## Tournament Simulator

The simulator forecasts each remaining team's chances of:

- **Win Title** — winning the entire tournament
- **Reach Final** — making it to the Final match (win or lose)

### How it works

1. **Exact QF enumeration** — all possible Quarterfinal outcomes are enumerated using real model match probabilities (not heuristics)
2. **Monte Carlo sampling (N=2000)** — for each simulated run, a QF outcome is drawn, then Semifinal and Final matches are resolved using real model predictions for every possible matchup pairing
3. **95% Confidence Intervals** — Wilson score intervals are computed across the 2000 runs, shown as ranges like `24–31%`

### Reading the simulator

```
        WIN TITLE   [████░░  ████░░]   REACH FINAL
France    28–36%                          50–58%
Spain     22–30%                          40–49%
```

- **WIN TITLE range** (`28–36%`) — the 95% CI for this team winning the tournament
- **Solid bar** — the model's point estimate
- **Faded bar extension** — the upper CI bound (uncertainty band)
- **REACH FINAL range** (`50–58%`) — the 95% CI for making it to the Final
- **SF ✓ badge** — team has already won their Quarterfinal

### Status line

Below the table:
```
1 confirmed · 3 pending · model predictions · N=2000
```

- `model predictions` — using real RF model outputs for all SF/Final matchups
- `strength heuristic` — fallback if hypothetical predictions aren't available yet

---

## Model

The underlying model is a Random Forest classifier trained on 964 historical
World Cup matches (1930–2022). It predicts three outcomes: home win / draw / away win.

**Key features:**
- Elo ratings (pre-match snapshot)
- Squad fingerprints (age, top-5 league share, diversity)
- Archetype labels (heavyweight clash, favorite vs underdog, host pressure, etc.)

**Validation (Leave-One-Tournament-Out cross-validation):**
- Log-loss: 0.9419
- Accuracy: 59.2%
- 2026 out-of-sample: 53.4% (73 matches)

---

## Data Updates

Match data is refreshed automatically every 6 hours via GitHub Actions. The
pipeline fetches live scores, updates Elo ratings, regenerates predictions,
and redeploys the frontend.

---

## Archetype Labels

Each match is classified with one or more structural archetypes:

| Archetype | Description |
|---|---|
| Heavyweight Clash | Both teams Elo > p75 |
| Favorite vs Underdog | Large Elo gap |
| Host Pressure | USA / Canada / Mexico involved |
| Generational Transition | Squad age profile contrast |
| Club Power Mismatch | Top-5 league share gap |
| Tactical Contrast | League diversity gap |
| Knockout Volatility | High-variance form profile |
