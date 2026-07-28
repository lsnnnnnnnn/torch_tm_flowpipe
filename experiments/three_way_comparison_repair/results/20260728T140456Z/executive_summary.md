# Executive summary

The repair selects **Outcome B**.

The historical three-way ranking is invalid. The generated Flow* adapter overwrote the native refined remainder after every successful `advance`, and Torch's displayed endpoint used an additional fixed-time residual tightening that Flow* and DiffReach did not use.

Stock Flow* reproduces the Riccati under-enclosure: at h=0.01 its upper endpoint misses the analytic upper bound by 1.24236723825e-08. A regenerated full-Picard inclusion test rejects the remainder-only refinement image. The original Flow* Van der Pol benchmark nonetheless reaches T=10 with 290 segments, and both identical-settings generated harnesses reproduce its schedule.

Accordingly, the report publishes the semantics-corrected Torch versus DiffReach rows, keeps Flow* stock and diagnostics visible, and issues no three-way width ranking.
