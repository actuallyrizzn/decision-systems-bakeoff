.PHONY: verify smoke-score report help

help:
	@echo "make verify DATA=...   — gate lockfile hashes"
	@echo "make smoke-score       — score a tiny synthetic JSONL against smoke sst2"
	@echo "make report RUNS=runs  — merge *_summary.json"

verify:
	python scripts/verify_data.py --data "$(DATA)"

smoke-score:
	python tests/make_and_score_smoke.py

report:
	python scripts/make_report.py --runs "$(RUNS)" --out "$(RUNS)/report.json"
