# Prediction JSONL schema

One JSON object per line. UTF-8.

## Required

| Field | Type | Meaning |
|---|---|---|
| `gold` | string | Gold label name (menu key) |
| `choice` | string | Model’s chosen label name |
| `probabilities` | object | Map label → probability (should cover the menu) |

## Strongly recommended

| Field | Type | Meaning |
|---|---|---|
| `i` | int | 0-based index into frozen `test.tsv` |
| `id` | string | Row id from `test.tsv` |
| `seconds` | float | Wall seconds for that decision |
| `brier` | float | Optional; scorer recomputes if absent |
| `hit` | bool | Optional; scorer recomputes if absent |

## Example

```json
{"i":0,"id":"dev-0","gold":"positive","choice":"positive","hit":true,"brier":0.02,"seconds":0.41,"probabilities":{"negative":0.1,"positive":0.9}}
```

## Summary JSON

`scripts/run_arm.py` also writes `*_summary.json` with at least:

`arm`, `task`, `n_scored`, `n_planned`, `accuracy`, `brier`, `median_seconds`, `stopped`, `lockfile_bakeoff_id`.
