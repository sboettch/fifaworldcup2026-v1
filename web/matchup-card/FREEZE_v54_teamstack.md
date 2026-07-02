# Matchup Card Freeze: v54 Team Stack

Date: 2026-07-02

Frozen review URL:

```text
http://127.0.0.1:8879/web/matchup-card/index-v54-teamstack.html?v=54teamstack
```

Frozen files:

- `index-v54-teamstack.html`
- `app-v54-teamstack.js`
- `styles-v54-teamstack.css`
- `data/matchups-v54-teamstack.json`
- `data/matchups-inline-v54-teamstack.js`

Frozen behavior:

- Section 1 uses status-first navigation: All 2026, Upcoming, Past results.
- Tournament context shows initial group, group opponents, selected stage, last result, and next matchup.
- Country shortcuts show group/next-match context rather than `x in view / y total`.
- Section 3 match card stacks country names vertically as home / v / away.
- Long country names use one-line dynamic sizing to prevent overlap.

Notes:

- This is a frozen artifact for review. The editable working fork remains `index-section1-fork.html`, `app-section1-fork.js`, and `styles-section1-fork.css`.
- The frozen page points to the frozen data snapshot, so future updates to the live matchup data should not change this version.
