# Flybrain arm

In-repo implementation under `src/decision_bakeoff/flybrain/` — frozen Science 2023 larva connectome, GloVe projection, sparse-CSR reservoir, ridge + temperature head. Hyperparameters locked in `configs/flybrain_best.json` (validation-grid winners from the published jevlab runs).

## Run (canonical for this bakeoff)

```bash
# once: GloVe 6B 100d (~822 MB extracted)
python scripts/fetch_glove.py --dir vectors

export BAKEOFF_DATA=~/path/to/locked/jevlab/data
python scripts/verify_data.py --data "$BAKEOFF_DATA"

python scripts/run_flybrain.py \
  --data "$BAKEOFF_DATA" \
  --glove vectors/glove.6B.100d.txt \
  --out runs/flybrain \
  --task sst2
```

Writes `runs/flybrain/flybrain_<task>_summary.json` and `*_rows.jsonl`. Re-score with `score_predictions.py` if you want an independent pass over the JSONL.

### Smoke (mini GloVe + fixtures; not locked hashes)

```bash
python scripts/run_flybrain.py \
  --data fixtures/smoke \
  --glove fixtures/smoke/glove.mini.txt \
  --out runs/smoke-fly \
  --task sst2 \
  --limit 4 \
  --skip-verify
```

## What stays frozen

| Piece | Rule |
|---|---|
| Connectome layout | Bundled `flybrain/data/*.csv.gz` — who connects to whom |
| Test rows | `lockfile.json` sha256 via `verify_data.py` |
| Arm hyperparameters | `configs/flybrain_best.json` per task |
| Train/val | Fit ridge λ + temperature only; never tune on test |

## Provenance

- Body: Winding et al., *Science* 2023 larva connectome (CC-BY via Netzschleuder)  
- Stack: vendored from [fly-cast](https://github.com/actuallyrizzn/fly-cast) jevlab / lab2 (this repo: AGPL code / CC-BY-SA docs — see `LICENSING.md`)  
- Hardware reference for published ms: CPU laptop, no GPU  

Do not describe a transformer fine-tune as “Flybrain” under bakeoff id `flybrain-jev-laya-2026-09`.
