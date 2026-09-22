# Flybrain arm

This bakeoff’s Flybrain arm is a **ridge classification head** on features from the published fruit-fly larva connectome (Winding et al., *Science* 2023). The graph stays frozen. Train/val fit the head; test is scored once.

## Option A — fly-cast (canonical implementation)

1. Clone and install https://github.com/actuallyrizzn/fly-cast  
2. Follow `tools/jevlab/README.md` / `docs/jevlab/PROTOCOL.md` for the readout recipe (grid on validation, never touch test during tuning).  
3. Export or convert scored test predictions into this repo’s JSONL schema (`docs/PREDICTION_SCHEMA.md`).  
4. Score:

```bash
python scripts/score_predictions.py \
  --data "$BAKEOFF_DATA" \
  --task sst2 \
  --arm flybrain \
  --predictions path/to/flybrain_sst2_rows.jsonl
```

Record the fly-cast commit SHA in your run notes.

## Option B — bring your own connectome readout

Any system is allowed if it:

1. Uses the **same locked** `test.tsv` (verify_data green)  
2. Uses the **same** decision menu / label names  
3. Emits prediction JSONL with probabilities over the full menu  
4. Documents training: what saw train/val vs test  

Then `score_predictions.py` is the scorer of record for this bakeoff repo.

## What “Flybrain” meant in the published piece

- Body: Science 2023 larva connectome (~3k neurons / ~117k edges)  
- Text → GloVe → reservoir on frozen wiring → pooled features  
- Thin ridge head + temperature on validation  
- Hardware reference: CPU laptop, no GPU required for the published run  

Do not describe a transformer fine-tune as “Flybrain” under this bakeoff id.
