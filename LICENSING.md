# Licensing — decision-systems-bakeoff

| Material | License | Paths |
|----------|---------|-------|
| Bakeoff orchestration (lockfile, Jev/Laya runners, scoring, docs except flybrain) | **MIT** | Root `LICENSE`, most of `src/decision_bakeoff/*.py`, `scripts/`, `docs/PROTOCOL.md`, … |
| **Flybrain arm** (vendored connectome stack) | **AGPL-3.0-or-later** | `src/decision_bakeoff/flybrain/**/*.py` (from [fly-cast](https://github.com/actuallyrizzn/fly-cast)) |
| Larva connectome CSVs | **CC-BY** (Winding et al. 2023 via Netzschleuder) | `src/decision_bakeoff/flybrain/data/*.csv.gz` — keep attribution |
| Flybrain data README | **CC-BY-SA-4.0** | `src/decision_bakeoff/flybrain/data/README.md` |

If you redistribute or modify the Flybrain package as a network service, AGPL corresponding-source obligations apply to that portion. The MIT bakeoff harness alone is not AGPL.

Copyright (c) 2026 Mark Hopkins / Decision Science Corp and contributors.
