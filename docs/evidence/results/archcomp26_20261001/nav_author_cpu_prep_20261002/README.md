# NAV CPU one-period preflight source snapshot

Executed on 2026-10-02 with `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python`. The script in this directory is a copy of the executed script; the editable source is [`tools/archcomp26_nav_author_cpu_preflight_nohash.py`](../../../../../tools/archcomp26_nav_author_cpu_preflight_nohash.py).

For each run, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Input models came from the previously saved official NAV preparation directory, `runs/archcomp26_20261001/nav_prep_001/official_nn-nav-{point,set}.onnx` on the server. The script used `--instance nav-standard` / `nav-robust`, corresponding to point/set; its `--output-dir` was a new sibling directory named by the run ID.

- [`nav_author_cpu_step1_standard_001`](../nav_author_cpu_step1_standard_001/RESULT.json)
- [`nav_author_cpu_step1_robust_001`](../nav_author_cpu_step1_robust_001/RESULT.json)

The `REFERENCE_CHECK.json` files record independent ONNX ReferenceEvaluator comparison. These records are center-point interface diagnostics only.
