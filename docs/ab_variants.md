# A/B Variants — Internal Notes

## Overview

The frontend ships two variants of the matchup tracker. They share the same
`app.js`, `styles.css`, and `data/matchups.json`. The only difference is the
`window.SIM_ENABLED` flag and the page title.

| File | Variant | Simulator | Title |
|---|---|---|---|
| `index.html` | B (default) | ✅ on | World Cup 2026 — Matchup Tracker |
| `index-a.html` | A | ❌ off | World Cup 2026 Tracker — Variant A |
| `index-b.html` | B | ✅ on | World Cup 2026 — Matchup Tracker |

`index.html` is always kept in sync with `index-b.html` so the root URL
defaults to the simulator-enabled experience.

---

## How the flag works

```js
// In index-a.html
window.SIM_ENABLED = false;

// In index-b.html / index.html
window.SIM_ENABLED = true;
```

In `app.js`, `renderSimulator()` checks at the top:

```js
if (window.SIM_ENABLED === false) {
  const card = simTable.closest(".audit-sim-card");
  if (card) card.hidden = true;
  return;
}
```

If the flag is `false`, the entire simulator card is hidden. If `true` or
unset, the simulator renders.

---

## Deployment

Both variants are deployed together to both repos:

| Repo | Visibility | URL |
|---|---|---|
| `sboettch/fifaworldcup2026` | Private | Internal only |
| `sboettch/fifaworldcup2026-v1` | Public | `worldcup26tracker.com` |

The public repo contains only the web frontend files (`web/matchup-card/`),
not the pipeline source code or raw data.

Files synced to public on every push:
- `web/matchup-card/index.html`
- `web/matchup-card/index-a.html`
- `web/matchup-card/index-b.html`
- `web/matchup-card/app.js`
- `web/matchup-card/styles.css`
- `web/matchup-card/data/matchups.json`
- `docs/`

---

## Testing locally

```bash
# Start local server
python3 -m http.server 8765 --directory web/matchup-card

# A variant (no simulator)
open http://localhost:8765/index-a.html

# B variant (with simulator)
open http://localhost:8765/index-b.html
```

---

## Adding a new variant

1. Copy `index-b.html` to `index-c.html`
2. Set `window.SIM_ENABLED` or add any other feature flags
3. Add to the sync block in the push script
