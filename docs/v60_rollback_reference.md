# v60 Rollback Reference

**Tagged:** `v60-sim-provenance`  
**Date frozen:** 2026-07-14  
**Status:** ✅ Confirmed working on live site

---

## Git Tags

```bash
# Private repo
git checkout v60-sim-provenance   # in fifaworldcup2026

# Public repo
git checkout v60-sim-provenance   # in fifaworldcup2026-v1
```

## Commit Hashes

| Repo | Hash |
|---|---|
| Private (`fifaworldcup2026`) | `f889cce41562700f313e7073af99f8b44eaa456d` |
| Public (`fifaworldcup2026-v1`) | `e02058fbed92fda41b25786aaee0c70e6e7471e1` |

---

## Versioned Files

| File | Purpose |
|---|---|
| `web/matchup-card/app-v60-sim-provenance.js` | Main JS — confirmed working |
| `web/matchup-card/styles-v60-sim-provenance.css` | CSS snapshot |
| `web/matchup-card/styles.css` | Canonical stylesheet (what HTML actually loads) |

---

## To Restore the Live Site

If anything breaks after v60, restore with two steps:

**Step 1 — Swap the JS reference in all three HTML files:**
```python
import re
for f in ['index.html', 'index-a.html', 'index-b.html']:
    c = open(f).read()
    c = re.sub(r'app[^"]*\.js\?v=[^"]+', 'app-v60-sim-provenance.js?v=60a', c)
    open(f, 'w').write(c)
```

**Step 2 — Copy the snapshot JS over canonical:**
```bash
cp web/matchup-card/app-v60-sim-provenance.js web/matchup-card/app.js
cp web/matchup-card/styles-v60-sim-provenance.css web/matchup-card/styles.css
```

**Step 3 — Push:**
```bash
git add web/matchup-card/ && git commit -m "rollback: restore v60" && git push
```

---

## What v60 Contains

### Features locked in
- **SF result locking** — confirmed SF winners/losers baked into MC loop; eliminated teams (e.g. France after SF1) show 0% and are filtered from table
- **Bracket-aware "Most likely final"** — picks top team from SF1 half (QF1+QF2 winners) and SF2 half (QF3+QF4 winners) separately; can never pair two teams from same bracket half
- **Plain-English provenance note** — dynamic text below simulator explaining confirmed finalists, eliminated teams, pending SFs, and what CI bands mean
- **Uncertainty callout** — amber annotated card on upcoming KO matches where draw > 22% or margin < 8pp; hidden on completed matches, group stage, and dominant results
- **KO-adjusted odds** — draw probability split 50/50 for penalty estimation, shown in callout
- **Prediction entropy** in bits (max 1.585 for 3-outcome model)

### Bugs fixed in v60 (from v59)
- Missing `for` loop in Monte Carlo simulator — was causing `SyntaxError: continue outside loop` that broke entire page
- Missing closing `}` on simulator function — trapped all downstream functions inside it
- `[hidden]` CSS override — `display: flex` was overriding the HTML `hidden` attribute, showing empty amber box on non-triggering matches
- `hasResult` guard extended — `result_available_not_modeled` fixture status (Spain/Belgium, France/Morocco etc.) was not caught by `!!match.score` alone

### Known limitations of v60
- No penalty shoot-out model — draw probability split 50/50 is a neutral assumption
- Model predicts 90-minute outcomes only
- SF result locking uses `actual_score` field — requires `actual_available: true` in matchups JSON
- Canonical `styles.css` used instead of versioned CSS to avoid CDN 404 risk

---

## Version History

| Version | File | Feature |
|---|---|---|
| **v60** ← current | `app-v60-sim-provenance.js` | SF locks + provenance note + all fixes |
| v59 | `app-v59-uncertainty-callout.js` | Uncertainty callout (had MC loop bug) |
| v58 | `app-v58-multi-badge-animation.js` | Monte Carlo simulator + CI bands (last stable before v59) |
| v57 | `app-v57-badge-animation.js` | Badge animation |
| v56 | `app-v56-match-animation.js` | Match animation |
| v55 | `app-v55-archetype-animation.js` | Archetype animation |

> **Safe rollback target if v60 fails:** v58 (`app-v58-multi-badge-animation.js`) — stable before all simulator changes.
