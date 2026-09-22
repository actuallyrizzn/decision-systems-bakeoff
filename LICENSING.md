# Licensing — decision-systems-bakeoff

| Material | License | Typical paths |
|----------|---------|----------------|
| **Software / source code** | **AGPL-3.0-or-later** ([full text](licenses/AGPL-3.0.txt)) | `src/**/*.py`, `scripts/**/*.py`, `tests/**/*.py`, `Makefile`, packaging (`pyproject.toml` as build metadata for the code) |
| **Documentation and other non-code** | **CC-BY-SA-4.0** ([full text](licenses/CC-BY-SA-4.0.txt)) | `README.md`, `docs/**` (including markdown + charts under `docs/illustrations/`), `questions/**`, `lockfile.json`, `configs/**`, authored fixtures prose / smoke TSV banks, this file |

Root [`LICENSE`](LICENSE) is a pointer. Full texts sit in [`licenses/`](licenses/).

SPDX: code `AGPL-3.0-or-later`; docs `CC-BY-SA-4.0`.

## Why AGPL on the code

This bakeoff ships runnable decision arms. AGPL means a modified network service still owes users corresponding source.

## Why CC-BY-SA on docs and fixtures

Prose, charts, locked question objects, and scoreboard tables should remix with attribution and share-alike without dragging documentation under AGPL.

## Mixed packages

Executable / library source → AGPL. Narrative, JSON question menus, TOML/JSON configs, markdown, illustration PNGs → CC-BY-SA. When in doubt, split by portion.

## Third-party material

| Artifact | Upstream | Notes |
|----------|----------|--------|
| `src/decision_bakeoff/flybrain/data/*.csv.gz` | **CC-BY** (Winding et al. 2023 via Netzschleuder) | [data README](src/decision_bakeoff/flybrain/data/README.md) — keep attribution; not relicensed by us |
| Benchmark corpora (SST-2, CLINC, MSR bug titles) | See `lockfile.json` / `docs/FETCH_DATA.md` | Not shipped here; retain upstream licences when you fetch them |
| GloVe 6B | Stanford / PDDL 1.0 (vectors) | Fetched by `scripts/fetch_glove.py`; not committed |

## Copyright

Copyright (c) 2026 Mark Hopkins / Decision Science Corp and contributors, unless a file says otherwise.
