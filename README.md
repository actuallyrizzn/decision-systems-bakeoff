# Decision systems bakeoff

**Reproduce the three-arm decision bakeoff:** Flybrain · Jev · Laya on the same frozen public rows.

This repo is the **prescription** — locked questions, locked test checksums, scoring contract, and runners for the hosted (Jev) and open-weight (Laya) arms. The Flybrain arm is documented here and implemented in [`fly-cast`](https://github.com/actuallyrizzn/fly-cast); you can also drop any system’s predictions as JSONL and re-score.

Published writeup (context, not required to run): Decision Science Corp blog *Same questions, three decision systems*.

---

## What you are reproducing

| Rule | Meaning |
|---|---|
| Same rows | Four frozen `test.tsv` files; sha256 in `lockfile.json` |
| Same questions | Fixed choice objects under `questions/` (clinc150 built from `labels.json`) |
| Same KPIs | Accuracy, Brier, median seconds, finishability (`n_scored / n_planned`) |
| Cold vs sticky-note | Jev and Laya: **no** task training. Flybrain: ridge head on **train** only; connectome wiring frozen |

Arms:

1. **Jev** — Venice `jev-latest` via `POST /api/v1/decisions`
2. **Laya** — `convaiinnovations/laya` on your machine (multilingual long-window for clinc150)
3. **Flybrain** — larva connectome readout (see `docs/FLYBRAIN.md`)

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1) Point at frozen data (rebuild: docs/FETCH_DATA.md)
export BAKEOFF_DATA=~/fly-cast-runs/jevlab/data   # example path

# 2) Refuse to proceed if checksums drift
python scripts/verify_data.py --data "$BAKEOFF_DATA"

# 3a) Jev arm (needs Venice key)
export VENICE_JEV_AB_API_KEY=…   # or VENICE_API_KEY
python scripts/run_arm.py --data "$BAKEOFF_DATA" --out runs/jev --task sst2 --arm jev

# 3b) Laya arm (needs `pip install laya` / ConvAI package)
python scripts/run_arm.py --data "$BAKEOFF_DATA" --out runs/laya --task sst2 --arm laya

# 3c) Flybrain — produce prediction JSONL (fly-cast), then:
python scripts/score_predictions.py \
  --data "$BAKEOFF_DATA" --task sst2 --arm flybrain \
  --predictions runs/flybrain/flybrain_sst2_rows.jsonl

# 4) Merge summaries
python scripts/make_report.py --runs runs --out runs/report.json
```

### Smoke (no locked corpora)

Smoke fixtures under `fixtures/smoke/` are **not** the locked bakeoff hashes. Use only to exercise plumbing:

```bash
python scripts/run_arm.py \
  --data fixtures/smoke --out runs/smoke --task sst2 --arm jev \
  --limit 2 --skip-verify
```

---

## Layout

| Path | Role |
|---|---|
| `lockfile.json` | Bakeoff id, arm definitions, per-task row counts + **test sha256** |
| `questions/` | Frozen decision objects (byte-stable menus) |
| `scripts/verify_data.py` | Gate: mismatch hash → exit 1 |
| `scripts/run_arm.py` | Jev / Laya runners → `*_summary.json` + `*_rows.jsonl` |
| `scripts/score_predictions.py` | Score any arm’s JSONL against gold |
| `scripts/make_report.py` | Merge summaries |
| `docs/PROTOCOL.md` | Human protocol (do not edit mid-run; amend with a dated note) |
| `docs/PREDICTION_SCHEMA.md` | JSONL row schema |
| `docs/FLYBRAIN.md` | How to run / import the connectome arm |
| `docs/FETCH_DATA.md` | How to rebuild the frozen corpora |
| `docs/PUBLISHED_RESULTS.md` | Our Sept 2026 reference scoreboard |

---

## Prediction contract

Every arm must emit (or be convertible to) JSONL rows:

```json
{"i": 0, "id": "…", "gold": "positive", "choice": "positive", "probabilities": {"negative": 0.1, "positive": 0.9}, "seconds": 0.42}
```

See `docs/PREDICTION_SCHEMA.md`. Brier and accuracy are recomputed from `choice` / `probabilities` / `gold` when you use `score_predictions.py`.

---

## Honesty clauses (do not strip)

- Jev and Laya are **cold** in this protocol.
- Flybrain trains a **thin readout** on the public train split; the connectome graph stays frozen.
- clinc150 Jev may stop early if the hosted API dies — report `n_scored` / `n_planned` (finishability).
- Matching `docs/PUBLISHED_RESULTS.md` is optional. Matching **lockfile hashes + question objects** is mandatory for a valid comparison.

---

## License

Code: MIT (see `LICENSE`). Datasets retain their upstream licences (see lockfile + `docs/FETCH_DATA.md`).
