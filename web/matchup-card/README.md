# Matchup Card Web App

Small static app for browsing the repo's 2026 World Cup matchup predictions as
archetype cards.

The UI uses full country names and plain-language tournament phases. For
example, raw `Group A` data is shown as `Opening round - Section A` so the app
does not assume the reader already knows tournament jargon.

The primary interaction is a chronological 2026 fixture board. Green cards are
exact model-correct rows, red cards are clear misses, amber cards are actual
draws where a side call receives partial credit, and teal/green pending cards
are forecast rows awaiting an actual result. Selecting any card updates the
match card, probability panel, radar comparison, archetypes, timing, upset risk,
and feature details.

## Regenerate Data

```bash
make matchup-card-data
```

This writes:

```text
data/processed/2026_fixture_status.csv
data/processed/2026_fixture_status_summary.json
data/processed/predictions_2026_scheduled.csv
data/processed/2026_matchup_viz.csv
data/processed/2026_matchup_viz_summary.json
web/matchup-card/data/matchups.json
```

The JSON is built from processed repo artifacts:

- `data/processed/predictions_2026.csv`
- `data/processed/predictions_2026_scheduled.csv`
- `data/processed/2026_matchup_viz.csv`
- `data/processed/2026_matchup_viz_summary.json`
- `data/processed/2026_fixture_status.csv`
- `data/processed/2026_fixture_status_summary.json`
- `data/processed/validation_2026.csv`
- `data/processed/fact_matchup_features.csv`
- `data/processed/fact_matchup_features_2026_scheduled.csv`
- `data/processed/fact_matchup_archetype.csv`
- `data/processed/fact_matchup_archetype_2026_scheduled.csv`
- `data/processed/cluster_assignments.csv`
- `data/processed/feature_importance.csv`

## Run Locally

From the repo root:

```bash
python -m http.server 8765
```

Open:

```text
http://localhost:8765/web/matchup-card/
```

Rows are true pre-match forecasts only when `pre_match_snapshot` is true. The
current local 2026 rows are retrospective validation artifacts unless the
scheduled-forecast layer marks them as `pre_match_forecast`.

`data/processed/2026_matchup_viz.csv` is the app-facing contract. It has one row
per announced fixture and keeps prediction fields separate from actual-result
fields. Pending fixtures use `is_imputed_prediction = True`, `actual_available =
False`, and `display_result_source = prediction_imputed`; as the live collector
ingests a result, the actual columns fill without overwriting the forecast
history. The same table also records `predicted_at_utc`, `prediction_timing`,
`training_split_name`, `training_cutoff_date`, `evaluation_split_name`, and
`audit_role` so the app can separate retrospective validation rows from
pre-match forecasts awaiting actuals. The grading layer keeps strict
`prediction_exact_correct` available, while `prediction_grade` and
`prediction_credit` support the draw-adjusted app score.

Announced fixtures outside the main validation layer are also scored by
`predictions_2026_scheduled.csv`. The UI labels those rows as pre-match
forecasts, post-kickoff pending-result forecasts, or retrospective scheduled
forecasts depending on kickoff time and result availability.
