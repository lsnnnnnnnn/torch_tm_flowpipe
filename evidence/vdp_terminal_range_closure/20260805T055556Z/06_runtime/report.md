# Runtime summary

At T=1 the proactive 4-leaf lane took 25.18994962517172 s versus 24.63779387716204 s for natural (1.0224x). The failed fresh T=10 request took 315.26696055568755 s and reached 6.397083942944808.

The synchronized eager microbenchmark covers CPU/CUDA, batch 1/16/48, and 4/16/64 leaves. All 18 rows are finite and coverage-valid. At batch 1 / 4 leaves the steady median is 2.691 ms CPU and 6.415 ms CUDA; at batch 48 / 64 leaves it is 269.226 ms CPU and 564.302 ms CUDA. CUDA leaf evaluation is faster in the latter row (2.246 ms versus 33.933 ms), but owner-cover construction and independent coverage validation remain host-oriented, so no end-to-end GPU speedup is claimed. The path is eager; compile time is explicitly not applicable, while setup, first call, warm-up, and steady timing are separate.
