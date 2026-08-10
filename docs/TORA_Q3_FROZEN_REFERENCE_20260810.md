# TORA Q3 frozen reference

Date: 2026-08-10

The latest reviewed Xiangru source is the server-resident
`origin/2026_experiment` reference
`5a3f94b28a7303c42a34fb6d57ebdaba63f25e42`.  The frozen Torch delivery is
`codex/tora-q3-stage-parity-fused-kernel-native-t20-20260809` at
`c60539f078e637b1169792d291f17afbe0a2a0fb`.  Neither lineage is merged into
this general VDP branch.

Valid reusable evidence is limited to the methodology: complete-Q3-K2 + DR-RP
can verify homogeneous TORA T20 at B48; Flow* O3 verifies at B12; a compiled
fixed-shape implementation can reproduce coefficient/mask/verdict semantics;
cold, warm, core, and process timing must be separated; and decisive CUDA
arithmetic needs independent outward qualification when universal directed
rounding is unavailable.

Unresolved items include a universal CUDA outward-rounding proof, Torch's
fresh-process graph setup, and an algorithmic-work advantage over Flow*.
Fixed31 did not deliver a runtime advantage; B26 failed at segment 186 after a
validated prefix through 18.5s.  Native DiffReach T20 remains incomplete.

Native T20 is no longer the project gate because TORA was a stress test, not
the objective.  The current gate is a generic PyTorch Taylor-model architecture
that reproduces DiffReach's fixed support, explains Flow*/Torch cross-step
divergence on VDP, and tests one sound generic carry improvement.  No
TORA-specific algorithm work, T20 rerun, B48 replay, controller optimization,
partition tuning, or property tuning is authorized in this round.
