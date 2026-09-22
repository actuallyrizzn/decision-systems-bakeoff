.PHONY: verify smoke-score smoke-fly report help

help:
	@echo "make verify DATA=...   — gate lockfile hashes"
	@echo "make smoke-score       — score a tiny synthetic JSONL against smoke sst2"
	@echo "make smoke-fly         — Flybrain smoke on fixtures (mini GloVe)"
	@echo "make report RUNS=runs  — merge *_summary.json"

verify:
	python scripts/verify_data.py --data "$(DATA)"

smoke-score:
	python tests/make_and_score_smoke.py

smoke-fly:
	python scripts/run_flybrain.py \
	  --data fixtures/smoke \
	  --glove fixtures/smoke/glove.mini.txt \
	  --out runs/smoke-fly \
	  --task sst2 \
	  --limit 4 \
	  --skip-verify

report:
	python scripts/make_report.py --runs "$(RUNS)" --out "$(RUNS)/report.json"
