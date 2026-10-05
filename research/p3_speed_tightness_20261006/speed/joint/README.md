# Optional joint QUAD candidate — prepared, not executed

`run_quad_joint_candidate.py` composes two unchanged adapters: the frozen sin/cos power-reuse adapter and anchored-control v2. This is one new numerical candidate, not a splice of the two parent results. No remote or numerical execution has been performed by this preparation task; the root scheduler decides whether the completed control-only result justifies it.

The inherited October 5 QUAD wrapper continues to own private output allocation, weighted256, order3/point2/validation4, original strict injection and endpoint processing, full SR history, every saved pooled tube/endpoint observation, memory cap, resource guard, and first-refusal stop. No fused weighted QUAD path is used.

The joint hook installs trig power after canonical preparation and before the production engine. It executes the unchanged new small-input trig GPU gate, then installs anchored control v2, then calls the original driver. Nested `finally` blocks restore control first and trig second, including failures in execution, recording or restoration. The inherited wrapper independently restores private output and weighted256 bindings and rejects final restoration failure.

`WIDTH_COMPARISON_BEFORE_JOINT_GATE.json` is saved before optimization qualification. `JOINT_QUALIFICATION.json` requires successful new trig gate and both restores, zero trig execution failures, positive opposite-op reuse, and every control refresh. Batch2 requires base2 + extra2 = actual4 NN calls and 6144 control rows. Full50 requires base50 + extra50 = actual100 NN calls, 50 injections and 153600 control rows. Counts for trig execution refer to Python eager/capture construction, not CUDA replay steps. The final width receipt is named `WIDTH_COMPARISON.json`; there is explicitly no saved-output-equality claim.

The inherited width comparison requires all 40/1000 steps and 1024 accepted lanes, saves every 12-state endpoint/tube bound and exact-rational width comparison (960/24000 rows), and compares control-refresh coverage to the saved reference. Canonical SR guards still require full history and only allow reset after length1000. The frozen reference remains read-only. The result is conditional on the two floating NN enclosures; independent end-to-end NNCS certification remains false.

The files are selected by explicit `--trig-dir` and `--control-dir`, not an ambient import path. The former must contain `trig_power_reuse.py` and `trig_power_gate.py`; the latter must contain the unchanged v2 `anchored_control.py` and `quad_controller_hook.py`. The hook's absolute `anchored_control` import is bound to that exact loaded file; an occupied module slot is refused. The control receipt must report `v2_real_support_exponent_lookup`.

After the root authorizes a new experiment and freezes these dependencies, the entry point is:

```sh
env CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=10 1800 taskset -c 14-17 \
  "$PYTHON_EXE" -B "$JOINT_SOURCE/run_quad_joint_candidate.py" \
  --mode full50 --base-wrapper "$QUAD_BASE" \
  --trig-dir "$TRIG_SOURCE" --control-dir "$CONTROL_V2_SOURCE" \
  --gate "$PRIVATE_GATE" --output "$NEW_OUTPUT"
```

`NEW_OUTPUT` must not exist and must be outside every source folder; the inherited supervisor must separately save outer START/RESULT/exit records. Use `--mode batch2` and a 180-second cap for a newly authorized short candidate. The default frozen paper source is retained; `--source-runner` may explicitly select an independently frozen compatible runner. This is an interface description, not permission to rerun old results or to launch now.

`check_qualification.py` uses only synthetic Python records to check complete 4/100-call coverage and rejection of missing extra NN calls, failed trig execution, missing restoration or missing GPU gate. It imports no Torch and runs no ODE/NN/GPU/checker. Its PASS is not numerical qualification of the joint path. AST and CLI-help checks also run without loading the saved runtime.
