# Fetch / freeze data

Locked test files are **not** vendored in this repo (licence + size). Rebuild them with the same recipes as the lab, then gate with `scripts/verify_data.py`.

## Recommended path (fly-cast)

The freeze scripts that produced the lockfile hashes live in:

https://github.com/actuallyrizzn/fly-cast → `tools/jevlab/`

| Script | Task |
|---|---|
| `fetch_sst2.py` | sst2 |
| `fetch_clinc.py` | clinc10 + clinc150 |
| `fetch_bugsev.py` | bugsev |

Typical layout after freeze:

```text
$data/
  sst2/test.tsv
  clinc10/test.tsv + labels.json
  clinc150/test.tsv + labels.json
  bugsev/test.tsv
```

Then:

```bash
python scripts/verify_data.py --data "$data"
```

Every hash must print `OK`. If any `FAIL`, you are not on the bakeoff rows — stop.

## Sources (also in lockfile)

- SST-2: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip  
- CLINC: https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json  
- Bugsev: https://github.com/ansymo/msr2013-bug_dataset  

Respect upstream licences. Cite Lamkanfi, Perez, Demeyer (MSR 2013) for bugsev.

## Smoke fixtures

`fixtures/smoke/` is for plumbing tests only. It intentionally does **not** match lockfile sha256. Always pass `--skip-verify` when pointing runners at smoke data.
