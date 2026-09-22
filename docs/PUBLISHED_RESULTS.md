# Published reference results (Sept 2026)

Canonical narrative (charts + markdown tables): root **[`README.md`](../README.md)**.

Optional target numbers only — a valid reproduction does **not** require matching these; it requires matching lockfile hashes and question objects.

## Accuracy

| Test | Rows | Flybrain | Jev | Laya |
|---|---:|---:|---:|---:|
| Movie reviews (sst2) | 872 | 76.5% | 94.6% | 90.6% |
| Ten intents (clinc10) | 300 | 94.7% | 99.3% | 97.3% |
| 150 + oos (clinc150) | 5500 | 64.6% | 91.7%† | 60.6% |
| Bug titles (bugsev) | 2000 | 78.4% | 49.1% | 29.5% |

† Jev: 4,060 / 5,500 before hosted API stopped answering.

## Median ms / row

| Test | Flybrain | Jev | Laya |
|---|---:|---:|---:|
| sst2 | 2.5 | 436 | 340 |
| clinc10 | 4.8 | 434 | 365 |
| clinc150 | 5.4 | 435 | 4063 |
| bugsev | 2.1 | 438 | 314 |

## Notes

- Jev metered spend for the published A/B ≈ **$0.48** at $0.042 / 1M input.
- On-prem arms measured on a ~$350 Lenovo IdeaPad Slim 3 (i3-N305, 8 GB, no GPU).
- Training honesty: Jev/Laya cold; Flybrain ridge head on train only.
