# Benchlab — MoEjev Phase 1

**Benchlab is the machine that produces the decision matrix.** MoEjev (Phase 2)
is a public router over that matrix; Phase 3 trains specialists into holes the
matrix proves. Nothing here is a vibes board — a cell exists because a locked
run produced it.

## What a bench IS

A bench is frozen and byte-stable **before** any arm touches it:

| Field | Rule |
|---|---|
| `task` id | snake_case, permanent (e.g. `agnews`) |
| source | URL + license, recorded |
| menu | the exact option texts the arm sees — on a decision model the menu *is* the test |
| rows | frozen test split, `rows_test` count |
| `test_sha256` | `scripts/verify_data.py` must pass before a run is "locked" |
| fidelity | `full` by default; `pilot-*` / `smoke` cells are labeled and never mixed into the full-fidelity board |
| KPIs | accuracy, brier, median_seconds, n_scored/n_planned, finishability, usd (hosted arms) |

Registry of record today: `lockfile.json` (suites `original` + `fair`). New
benches register there first — same shape, same freeze rules — then land in
`questions/` as byte-stable objects.

## The matrix

`runs/matrix.db` (SQLite, `src/decision_bakeoff/matrix.py`). One row per
(arm, task, run_id); re-ingesting a run updates in place. Cells carry fidelity
and host so a pilot never silently compares against a full run.

```bash
# ingest a run directory
python scripts/matrix_ingest.py runs/<run-dir> --run-id <run-dir> --suite fair --host ngram

# the board (latest full-fidelity cell per arm x task)
python scripts/matrix_ingest.py --board --suite fair
```

## The queue

`bench/queue.json` — ranked candidate nodes (new tasks, task extensions, new
arms, fidelity re-runs). Spare home compute pops the highest-value node, runs
it locked, ingests, marks done. The queue is the always-on consumer for idle
CPU; the matrix is the product.

## Phase gates

- **Phase 1 → 2:** the matrix shows real service savings on real traffic
  shapes (cheaper/faster/more finishable than always-one-arm), not a hunch.
- **Phase 2 → 3:** router revenue funds training specialists for cells the
  matrix already proved are worth owning.

## Hard rules

1. No cell without a locked bench (sha256-verified split, frozen menu).
2. No mixing fidelities on one board — filter or label.
3. Amendments are dated and appended (PROTOCOL.md rule); history is never rewritten.
4. Every run ingests into the matrix the day it finishes — a run that isn't in
   the DB doesn't count.
