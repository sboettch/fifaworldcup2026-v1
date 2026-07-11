# Model Uncertainty Callout

**Added in v59 · July 11 2026**

---

## What it is

For knockout matches where the model's confidence is low, a callout appears
in the Probability panel explaining the uncertainty in plain terms before
kick-off.

It does not change the model's prediction. It explains why you should not
be too confident in it.

---

## When it appears

The callout fires on **upcoming knockout matches only** when either:

- **Draw probability > 22%** — above the historical knockout-stage mean (~18%)
- **Margin < 8 percentage points** between the two teams

It is hidden on:
- All completed matches (regardless of stage)
- Group stage matches
- Matches where the model is confident (low draw, clear margin)

---

## What it shows

```
MODEL UNCERTAINTY NOTE
High draw probability — prediction confidence is low

Draw probability (X%) is above the historical KO-stage mean of ~18%.
A margin of Y pp between the two teams is within the model's typical
noise floor. The model pick is [Team] — but this is not a confident call.

In a knockout, a draw after 90 mins goes to extra time — and likely a
penalty shoot-out. Redistributing draw probability 50/50 gives
KO-adjusted advancement odds: [Home] X% · [Away] Y%.

Prediction entropy: X.XX bits / 1.58 max (Z% of maximum uncertainty)
· Margin: Y pp · 90-min outcomes only
```

---

## Why "90-min outcomes only"

This model is trained on 90-minute match results. It does not model:

- Extra time
- Penalty shoot-outs
- In-match events (red cards, injuries)

When a knockout match is predicted to draw (e.g. 26%), the true probability
of each team *advancing* is higher than their raw win probability because
they each have a ~50% chance of winning the shoot-out.

The KO-adjusted odds redistribute that draw probability 50/50 across
a hypothetical shoot-out. This is a neutral assumption — we do not have
a penalty shoot-out model.

---

## Example: France vs Spain, SF Jul 14

| | 90-min | KO-adjusted |
|---|---|---|
| France win | 35.3% | 48.3% |
| Draw | 25.9% | → split |
| Spain win | 38.8% | 51.8% |

Prediction entropy: 1.57 bits out of a maximum of 1.58 bits.
At maximum entropy all three outcomes would be equally likely (33.3% each).
This match is essentially at that ceiling.
