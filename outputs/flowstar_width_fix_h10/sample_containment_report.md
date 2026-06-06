# Sample Containment Report

This diagnostic checks deterministic RK4 sample trajectories against PyTorch normalized-insertion boxes at accepted segment end times.
It is a sanity check only, not a reachability proof.

## Result

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_insert_scalars | 500 | 75000 | 0 | 0.0 | passed |

Max outside sample id: ``.
Max outside time: ``.
Integrator: `rk4` with deterministic grid initial points in x=[1.1,1.4], y=[2.35,2.45].
Conclusion: this diagnostic can catch obvious enclosure misses, but passing it does not prove containment for the full initial box.
