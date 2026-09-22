# Decision systems bakeoff

**Reproduce the three-arm decision bakeoff:** Flybrain · Jev · Laya on the same frozen public rows.

This repo is the **prescription and the runners** — locked questions, locked test checksums, scoring contract, Jev/Laya arms, and an **in-repo Flybrain** arm (larva connectome + ridge head). You can also drop any system’s predictions as JSONL and re-score.

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
3. **Flybrain** — larva connectome readout (`scripts/run_flybrain.py`, `docs/FLYBRAIN.md`)

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

# 3c) Flybrain (once: python scripts/fetch_glove.py --dir vectors)
python scripts/run_flybrain.py \
  --data "$BAKEOFF_DATA" \
  --glove vectors/glove.6B.100d.txt \
  --out runs/flybrain \
  --task sst2

# 4) Merge summaries
python scripts/make_report.py --runs runs --out runs/report.json
```

### Smoke (no locked corpora)

Smoke fixtures under `fixtures/smoke/` are **not** the locked bakeoff hashes. Use only to exercise plumbing:

```bash
python scripts/run_arm.py \
  --data fixtures/smoke --out runs/smoke --task sst2 --arm jev \
  --limit 2 --skip-verify

python scripts/run_flybrain.py \
  --data fixtures/smoke --glove fixtures/smoke/glove.mini.txt \
  --out runs/smoke-fly --task sst2 --limit 4 --skip-verify
```

---

## Layout

| Path | Role |
|---|---|
| `lockfile.json` | Bakeoff id, arm definitions, per-task row counts + **test sha256** |
| `configs/flybrain_best.json` | Locked Flybrain hyperparameters per task |
| `questions/` | Frozen decision objects (byte-stable menus) |
| `src/decision_bakeoff/flybrain/` | Connectome + reservoir + ridge arm (AGPL) |
| `scripts/verify_data.py` | Gate: mismatch hash → exit 1 |
| `scripts/run_arm.py` | Jev / Laya runners → `*_summary.json` + `*_rows.jsonl` |
| `scripts/run_flybrain.py` | Flybrain runner |
| `scripts/fetch_glove.py` | Download GloVe 6B 100d |
| `scripts/score_predictions.py` | Score any arm’s JSONL against gold |
| `scripts/make_report.py` | Merge summaries |
| `docs/PROTOCOL.md` | Human protocol (do not edit mid-run; amend with a dated note) |
| `docs/PREDICTION_SCHEMA.md` | JSONL row schema |
| `docs/FLYBRAIN.md` | Flybrain arm details |
| `docs/FETCH_DATA.md` | How to rebuild the frozen corpora |
| `docs/PUBLISHED_RESULTS.md` | Our Sept 2026 reference scoreboard |
| `LICENSING.md` | MIT harness vs AGPL flybrain vs CC-BY data |

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

Orchestration: MIT (see `LICENSE`). Flybrain package: AGPL-3.0-or-later. Connectome CSVs: CC-BY. Details: `LICENSING.md`. Datasets retain their upstream licences (see lockfile + `docs/FETCH_DATA.md`).
