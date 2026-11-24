# Project handoff · 24 November 2026

This project computes enclosures of the trajectories of a physical system controlled by a neural network. The selected Unicycle implementation completed the schedule in 4.30 seconds, compared with 4.80 seconds for Huan, and attained the smallest endpoint and maximum tube width on each of the four physical states among the methods admitted to that comparison. The less-robust Double Pendulum also had narrower bounds on all four states, at a higher time cost. Quadrotor completed all 1,000 steps at the selected degree and remained slower and generally wider than Huan. These measurements describe the selected implementations.

- Present the [English slides](slides.pdf) using the [speaker script](speaker_script.docx). The editable [Beamer source](slides.tex) and the [legacy Word version](speaker_script.doc) accompany them. Run `xelatex slides.tex` twice to rebuild the slides.
- Consult the [numerical source guide](../../research/handoff_method/README.md) for the controller, validation and history operations, their dependencies and the scope of the local check.
- Use the [selected results](results.json) to look up all 24 instances and 114 physical state entries. Each case gives the selected configuration, completion, run identifiers, timing boundary, number of samples, per-method width counts and individual widths when available.

## Recheck one result

From the project root, run the following with Python 3. It reads the published summary. Experimental reruns additionally require the matching plant, controller file, native extensions, device environment and original run configuration.

```sh
python3 - <<'PY'
import json
from pathlib import Path
records = json.loads(Path('docs/handoff/results.json').read_text())
case = next(c for c in records['cases'] if c['case_id'] == 'double-pendulum-less-robust')
print(case['our_selection'])
for method, time in case['time'].items():
    counts = case['smallest_width_counts'][method]
    print(method, time['seconds'], time['boundary'], counts['table_display'])
for state in case['state_widths']:
    print(state['state'], state['endpoint_widths'])
PY
```

The retained October 27 selection completed 20 numerical schedules. Continuous Airplane, Discrete Airplane, Cart-pole and the more-robust Double Pendulum have partial or diagnostic records. The completion count measures numerical progress. Read the case-specific safety monitor or counterexample record separately. Compare widths on the admitted horizon and task. Missing values and method exclusions retain their source labels.

The three representative timing rows in the slides carry different measurement histories. Our Double Pendulum and Quadrotor driver times use a production mode with reporting disabled. Our Unicycle uses recorded confirmations. The historic Flow* measurements include process capture. Each comparison keeps the timing boundary and property scope attached to the relevant row.

## Continuing the numerical work

Start with the less-robust Double Pendulum and record the exact order triple, controller, initial boxes, step size, reporting mode and driver timing boundary. The selected run used zero weighted maps. A single-mechanism study needs its matched comparator and the corresponding run receipts. For Quadrotor, the degree-matched route stopped at step 601. The later selected configuration completed the full schedule. Inspect each run's actual validation counters before attributing any change in time or width to a mechanism.

## Repository history

The project tree follows the October 6 published research commit `f1e5ade` with the October 27 result selection and a curated method handoff. The [former branch tips](branch-history.txt) map each historical name to its commit. You can inspect one with `git show <commit>` or `git ls-tree <commit>`. The reconciliation commit keeps the previous research branches reachable from main, including the divergent histories. Their experimental trees are available by commit.
