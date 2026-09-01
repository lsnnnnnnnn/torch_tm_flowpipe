# C4 independent CPU batch contract

The CPU batch foundation is an ordered collection of independently owned B1
polynomial-plant solver lanes. It establishes semantics for B=1, B=2, and B=8;
it does not claim a fused full-solver tensor kernel or GPU speedup.

Each `CPUPolynomialPlantLane` owns its current Taylor vector (including
polynomial coefficients and ordinary remainder), normal-flowpipe state,
accepted-boundary SR queue, owner/generation metadata, accepted/rejected
counters, refinement count and stop reason, frozen status, and checkpoint
fingerprint. Lane identifiers are unique and batch order is fixed.

An accepted result atomically replaces only that lane's current TM and normal
state. A rejected result increments only that lane's rejected count, freezes
the lane, and retains its prior current TM, queue, boundary, and generation
byte-for-byte. Frozen lanes are not called again. A lane's range policy and
stop-ratio decision never reduce or terminate another lane.

`save_cpu_batch_checkpoint` writes one existing safe terminal checkpoint per
lane plus a canonical batch manifest. Loading verifies the manifest checksum,
the declared contract, every terminal checkpoint, and every stored lane
fingerprint. A partial, reordered, stale, or tampered manifest fails closed.

The formal evidence covers:

- duplicate B1 embedding into B1, B2, and B8, with exact endpoint, tube,
  remainder, queue, replay, and status equality;
- a heterogeneous B8 containing the frozen accepted case, a shifted valid
  box, a designed early rejection, a valid box with a different refinement
  count, and valid duplicates;
- whole B8 versus 2×B4, 4×B2, and 8×B1 chunk invariance;
- B8 checkpoint/resume versus uninterrupted execution;
- diagnostic 8×serial-B1 and B8 wall time, throughput, and process peak RSS.

The formal clean-SHA run produced 51 equivalence rows with no failure. B8 took
9.249690 s versus 9.317899 s for 8×serial B1, a 0.992680× ratio and therefore
inside the required “not slower than 2×” diagnostic bound. The committed
`cpu_batch_equivalence.csv` and `cpu_batch_runtime.csv` contain the formal clean
batch-SHA run.

CUDA remains out of scope. The next CUDA lane must use these per-lane CPU
fingerprints as its oracle and define a separate floating-point soundness
contract before any controller integration.
