# First-order follow-up: correctness and matched bases

This experiment follows the frozen
`first_order_three_way/results/20260723T173852Z` benchmark.  It separates:

1. the basis used to construct a local segment;
2. the representation carried between segments; and
3. the method used to validate the resulting enclosure.

The copied `benchmark_spec.yaml` is a frozen input.  The historical result
directory is never written by these scripts.

Focused checks:

```bash
./experiments/first_order_followup/run_smoke.sh
```

Focused full run:

```bash
./experiments/first_order_followup/launch_background.sh
```

The full runner creates a fresh timestamped directory below `results/`.

Protocol A remains the frozen historical artifact.  This directory implements
Protocol B (`matched_affine_carry`), Protocol C
(`complete_degree_two_reference`, with DiffReach explicitly restricted), the
Flow* extraction gate, the Torch dependency audit, and the B1/B_DR/B2 ablation.

Key audit notes:

- [FLOWSTAR_EXTRACTION_AUDIT.md](FLOWSTAR_EXTRACTION_AUDIT.md)
- [TORCH_DEPENDENCY_AUDIT.md](TORCH_DEPENDENCY_AUDIT.md)
- [DIFFREACH_AFFINE_PROJECTION.md](DIFFREACH_AFFINE_PROJECTION.md)
- [MATCHED_BASIS.md](MATCHED_BASIS.md)

Run a full sweep directly when background execution is not desired:

```bash
./experiments/first_order_followup/run_all.sh /tmp/followup-result
```

Every curated result contains common raw/summary CSV files, correctness and
environment JSON, generated C++ sources/logs, eight required plots, and an
evidence-classified report.
