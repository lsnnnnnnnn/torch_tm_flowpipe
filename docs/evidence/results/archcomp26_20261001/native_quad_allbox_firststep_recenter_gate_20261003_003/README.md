# Isolated paper QUAD: all 1,024 boxes, first control call and first ODE step only

This is a **new** run at
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003`.
It copies the Flow* library from the previously saved [first-box recenter gate](../native_quad_initial_recenter_repair_gate_20261003_002/README.md), which includes the two-site VAR-tail repair and the two-sided `Interval::toCenterForm` radius repair. The original library, original full50 run and prior gate were not modified or restarted. No content digest or hash verification was performed.

The copied [source](quad_allbox.cpp) and [observer](quad_gate_observer.h) retain the 2026 paper ODE, source `8×8×8×2×1×1` physical initial partition, one batched CROWN call, float32 coefficient transport, order 2, and fixed `h=0.005`. It checks every saved RPC input interval against the corresponding source initial interval before propagation. It then calls `author_matched::reach` **sequentially once per box** and stops on the first numerical refusal. [Source changes](source.patch), [observer changes](observer.patch), the capped [build](build.sh) and [run](run_gate.sh) scripts, and [build log](build.log) are saved. The sole run used GPU 2, CPU core 10 and port 5000, with a 1,080-second cap on the native process. The RPC server exited after the run.

The raw [RPC](run/rpc.json), [controller call count](run/controller_rpc.jsonl), [source boxes](run/source_boxes.csv), [per-box status](run/state.csv), [native ranges](run/ranges.bin), [composed x1/x2 octagon](run/octagon.csv), [terminal axes](run/terminal_axes.csv), [native stdout](run/stdout.log), [native stderr](run/stderr.log), [server log](run/server.log), and [exit code](run/native_exit_code.txt) are preserved. The native observer was **on only**, avoiding a second CROWN call. All 1,024 boxes returned `status=2` with one accepted 0.005 s step; no numerical first refusal occurred. The saved C++ elapsed time was 4.618 s for these 1,024 first steps, with no speed ranking or extrapolation. The full control period in the frozen source is 0.1 s, or 20 such steps; it was **not** executed here. The raw stdout marker `PROPERTY_NOT_CHECKED_FIRST_PERIOD_ONLY` is imprecise legacy wording in this diagnostic source and means only the first control call's first small step.

The independent [no-hash audit](audit_nohash.py) confirmed the full source Cartesian partition, one 1,024-box RPC call, all 12,288 saved RPC coordinate intervals enclosing their source intervals, finite ordered CROWN coefficients, 1,024 sequential accepted status rows, and 8,192 complete finite ordered octagon rows. The existing [range scanner](../../../../../tools/scan_archcomp26_native_ranges_nohash.py) found 1,024/1,024 finite ordered `h=0.005` records, with no endpoint outside its same-step tube component. Its [scan](run/SCAN.json) and the combined [audit](AUDIT.json) retain the counts and any failure location; this attempt has none. [RESULT.json](RESULT.json) is the concise outcome.

`status=2` is numerical completion of one small step. The QUAD target and full-time reach-and-remain property were **not checked**. The first-box independent plant inclusion argument from the prior gate does not extend automatically to these other 1,023 boxes. This run also does not independently certify CROWN, the actual neural network, the remaining 19 steps in the first control period, later controls, the 1,000 steps to `T=5`, or observer on/off equivalence. **The native octagon production gate remains CLOSED.**

From the repository root, the saved receipts can be checked without a solver or digest:

```bash
/opt/anaconda3/bin/python3 -B tools/scan_archcomp26_native_ranges_nohash.py \
  --input docs/evidence/results/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003/run/ranges.bin \
  --output /private/tmp/quad-allbox-firststep-scan.json --boxes 1024 --steps 1 --physical 12
/opt/anaconda3/bin/python3 -B docs/evidence/results/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003/audit_nohash.py
```
