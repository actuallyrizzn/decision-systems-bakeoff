# Fair-turf bakeoff (Julia-friendly rematch)

Suite id: **`julia-fair-turf-2026-09`** in `lockfile.json`.

These are the tests where Supersonic’s own card says Julia should look competitive:
short clear menus, described options, and typed decisions — not Banking77-scale
intent lists or weak-signal bug titles.

## Tasks

| Task | Rows (test) | Labels | Arms |
|---|---:|---:|---|
| `agnews` | 7600 | 4 topics | Julia, Jev, Laya, Flybrain |
| `emotion` | 2000 | 6 emotions | Julia, Jev, Laya, Flybrain |
| `massive_scenario_en` | 2974 | 18 scenarios (en only) | Julia, Jev, Laya, Flybrain |
| `typed_decisions` | 400 cases / 2000 Qs | choice / score / noul | Julia, Jev, Laya, Flybrain (zero-shot cosine) |

Original suite (`sst2`, `clinc10`, `clinc150`, `bugsev`) stays locked. Writeups should
include Julia’s poor showing there (**Doc #1423**) plus this fair rematch.

## Freeze

```bash
python scripts/freeze_fair_bakeoff.py --out /path/to/data/fair
python scripts/verify_data.py --data /path/to/data/fair --suite fair
```

Data is not vendored (`/data/` is gitignored). Digests live in `lockfile.json`.

## Run

```bash
# Julia cold chain (classification + typed)
python scripts/run_fair_chain.py \
  --data "$FAIR_DATA" --out runs/fair-julia --status runs/fair-julia/STATUS.json \
  --arm julia --model-dir /path/to/Julia-1

# Per-task Jev / Laya
python scripts/run_arm.py --data "$FAIR_DATA" --out runs/fair-jev --task agnews --arm jev
python scripts/run_arm.py --data "$FAIR_DATA" --out runs/fair-laya --task emotion --arm laya

# Flybrain (needs train/valid + GloVe; provisional hypers in configs/flybrain_best.json)
python scripts/run_flybrain.py --data "$FAIR_DATA" --glove vectors/glove.6B.100d.txt \
  --out runs/fair-fly --task agnews

# Typed only
python scripts/run_typed_decisions.py --data "$FAIR_DATA" --out runs/fair-typed \
  --arm julia --model-dir /path/to/Julia-1
python scripts/run_typed_decisions.py --data "$FAIR_DATA" --out runs/fair-typed \
  --arm flybrain --glove vectors/glove.6B.100d.txt
```

## Scoring / telemetry notes

Same fidelity as the original suite (`run_arm.py` / Julia chain):

- **KPIs:** accuracy, Brier, median seconds, `n_scored` / `n_planned`, **finishability**
- **Per-row:** `i`, `id`, `gold`, `choice`, `hit`, `brier`, `seconds`, full `probabilities`,
  plus Jev `input_tokens` / `output_tokens`
- **Summary meters:** `model`, `index_offset`, `input_tokens`, `output_tokens`, `usd`,
  `balance_usd_last`, `wall_seconds`, `finished_at`, both lockfile ids
- Typed-decisions: argmax over probabilities for choice / score / noul (Julia harness rule;
  do **not** round score expected index); case-level token meters attributed to the first
  question row; `by_type_accuracy` + parquet sha
- Flybrain configs for fair tasks are **provisional defaults** (not a fresh val grid).
- MASSIVE here is **en scenario only**; full 52-locale is optional follow-up.

## Locked additions (2026-10-05)

**Banking77** is now a sibling suite (`locked_additions` in `lockfile.json`):
full 77 intents, public gold, **no shortlist**. Freeze with
`scripts/freeze_banking77.py`. Julia is skipped (native max 20 options), same
as clinc150. Flybrain uses a ridge head on train/val like the other TSV tasks.
Decision Index is a separate official kit — never train or tune on it.


1. Report original-suite Julia cold results (weak).
2. Report fair-suite four-arm board.
3. Do not overclaim from 100-row pilots on the model card — we run full frozen tests.
