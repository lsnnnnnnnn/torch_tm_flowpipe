# Fixed-batch weighted fusion candidate — 2026-10-06

This directory extends the October 5 single-lane speed idea to the existing
NAV B=25 and paper QUAD B=1024 contracts. It does not change a remainder
formula, strict boundary injection, controller, SR history, observer, or saved
range. No server numerical experiment was launched while preparing these files.

`weighted_fused_batch.py` replays both original weighted interval maps in one
CUDA graph per fixed chunk: NAV uses 32 scratch rows; QUAD uses four 256-row
chunks. Every chunk copies its result and flags before the next replay. One
packed host read accepts the candidate only when every row passes both maps,
all original initial/finite/zero-containment preconditions hold, and neither
intersection is disjoint. Otherwise the original `refine_accepted` control
flow supplies the result and diagnostics, using the already-qualified fixed
scratch size. Failed-lane and recentered validation outside this refinement
wrapper remain untouched. A trace request uses the original control flow.

The first successful live input is also processed by the original refinement,
with the same fixed32/fixed256 scratch policy. Returned tensor bytes and every
statistics field must match. This one-time check is included in timing. Its
separate reference graph pool is synchronized and released once its owned
result has returned, before the byte comparison; its
temporary peak can overlap the fused pool, so the existing resource guard is
still required. Fallback graph pools receive the same release treatment. The
fused graph retains one support-signature entry at a time in
`eng._weighted_graphs`, so the frozen engine's SR-growth memory reclamation
continues to clear it before SR buffer expansion.

This preserves per-row operation order but changes scheduling from “round 1
across chunks, then round 2” to “both rounds of each chunk.” The frozen map
allocates its coefficient/cache/tail outputs per evaluation, and the selected
private-output kernel binding supplies owned kernel outputs. The new live-input
check and full saved-output comparison are required evidence for this change;
the CPU check alone is not GPU qualification or an end-to-end NN certificate.

## Local check

`check_fused_batch_control.py` uses real CPU Torch 2.2.2 with a small interval-map
fixture and the frozen original Python control flow. `CPU_CONTROL_CHECK.json`
records 14 passing cases over both batch configurations: success, first/second
round failure, an ineligible row, initial mismatch, a nonfinite input, and a bad
map. The QUAD cases place the exceptional row in the second chunk. The fixture
reuses static output/flag storage exactly as GraphCache does, and checks that
earlier chunks and public outputs retain their own bytes. It also checks input
immutability and installation/restoration ownership. There were no GPU calls,
old checker runs, or content digest operations.

The production install is restricted to the existing PyTorch 2.5.1 environment.
No CUDA extension is compiled or loaded by this adapter; it uses the already
prepared runtime and graph layer.

## Runner and qualification

`run_fused_batch_candidate.py` loads an explicit frozen October 5 wrapper and
replaces only its weighted install function. It retains the original execution,
first-refusal handling, full saved-output comparison, and restoration. The NAV
branch replaces `install_weighted32`; the QUAD branch replaces
`weighted_chunk256.install` after the inherited unused fixed128 binding is
retired. Files in the frozen source directory are never edited.

The new runner accepts `--instance nav --mode full600` or
`--instance quad --mode batch2|full50`, plus `--base-wrapper`, `--gate`, and
`--output`. Additional source/reference options pass to the original wrapper.
The output must be outside the frozen wrapper folder. QUAD relaxation is fixed
to same-slope and comparison to equivalent; widths are not the objective here.

The parent scheduler should run NAV full600 first, then QUAD batch2 (40 steps),
and decide separately about full50 (1000 steps). The inherited affinity checks
still require NAV GPU1/CPUs6–9 and QUAD GPU3/CPUs14–17. Launch with `python -B`
and `PYTHONDONTWRITEBYTECODE=1`, in a fresh unique output directory, under the
unchanged supervisor/time/resource limits. This README is not a launch receipt.

Success requires all of:

- `refine_calls = fast_calls = expected steps`, zero fallbacks and candidate
  exceptions, exactly one reference check, and `reference_map_evaluations=2*B`;
- exact replay/map/real-row/padded-row counts for the selected fixed batch;
- all original saved numerical outputs and final fields match the original
  wrapper's archived reference (NAV all 25×600 ranges; QUAD all 40/1000 saved
  pooled tube/endpoint rows, not an unrecorded per-lane TM identity claim);
- complete original binding restoration and a successful supervisor exit.

`FUSED_BATCH_BINDING.json`, `FUSED_BATCH_QUALIFICATION.json`, and augmented
START/RESULT records identify the new adapter and counters. Numerical refusal,
resource failure, fallback, reference mismatch, or saved-output mismatch must
not be classified as a speed qualification. Full-process and internal-driver
times remain distinct; one new sample is not a repeated timing distribution.
