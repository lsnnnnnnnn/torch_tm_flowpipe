# flowstar_gpu developer workflow. No CI exists; `make check` is the pre-commit gate.

.PHONY: oracle oracle-data golden test test-full soak cov parity soundness adversarial adversarial-full bench bench-report check clean nncs-deps

# --- C++ oracle -------------------------------------------------------------

oracle:
	$(MAKE) -C tools/oracle

# Regenerates ALL Flow* ground-truth dumps (deterministic; ~1 min).
oracle-data: oracle
	scripts/gen_oracle_data.sh

# Extracts the small committed golden slices from oracle_data into tests/golden/.
golden:
	uv run python scripts/make_golden.py

# --- test suites ------------------------------------------------------------

# Fast dev loop: unit + property tests at the 'dev' hypothesis profile (~1 min).
test:
	uv run pytest tests/unit tests/properties -q -m "not slow"

# The 100%-coverage gate (~3 min). Requires CUDA (device-parametrized kernel tests).
cov:
	uv run pytest tests/unit tests/properties tests/parity -q -m "not slow" \
	    --cov --cov-report=term-missing

# Thorough run: 'ci' hypothesis profile (more examples per property).
test-full:
	HYPOTHESIS_PROFILE=ci uv run pytest tests -q -m "not slow" --cov --cov-report=term-missing

# Long fuzzing soak (run before milestone tags).
soak:
	HYPOTHESIS_PROFILE=soak uv run pytest tests/properties -q

# Full-horizon parity vs oracle_data + regenerate the PARITY.md result tables.
parity:
	uv run pytest tests/parity -q
	uv run python scripts/parity_report.py

# Monte-Carlo trajectory containment (soundness, independent of Flow*).
soundness:
	uv run pytest tests/soundness -q

# Adversarial soundness suite: PGD-optimized worst cases, pointwise TM
# containment, region splitting, mutation detector-power, barely-unsafe
# safety/NNCS specs, batch-lane invariance. Fast subset; adversarial-full
# adds the slow sweeps (256-cell splits, chaotic benchmarks).
adversarial:
	uv run pytest tests/adversarial -q -m "not slow"

adversarial-full:
	uv run pytest tests/adversarial -q

# --- benchmarking -----------------------------------------------------------

bench:
	uv run python bench/run_bench.py

bench-report:
	uv run python bench/make_report.py

# --- gate -------------------------------------------------------------------

check: cov
	uv run ruff check src tests scripts bench

# auto_LiRPA for the M7 NNCS demo: installed OUTSIDE pyproject because its
# setup pins torch<=2.11 (would fight our cu126 2.13 pin). Per user directive
# (2026-07-26) we use the INTERNAL development tree, which carries bug fixes
# ahead of the public release; verified against torch 2.13.0+cu126 (CROWN +
# return_A linear bounds bitwise-match the public 0.7.2 on the smoke test).
nncs-deps:
	uv pip install -p .venv/bin/python appdirs tqdm graphviz ninja onnx onnx2pytorch "onnxruntime>=1.28"
	uv pip install --no-deps -p .venv/bin/python -e /home/huan/projects/Verifier_Development

clean:
	$(MAKE) -C tools/oracle clean
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
