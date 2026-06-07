# Flowstar Linear Symbolic Queue V2 Notes

## What Changed

The old Stage-3 noise prototype introduced explicit symbolic noise variables into the Taylor-model polynomial state and later materialized them back into ordinary remainders. It was useful for testing dependency retention, but it did not model Flow*'s J/Phi_L/scalar queue structure.

The `normalized_insertion_symqueue_split` mode moved closer to the desired semantics by keeping the ordinary normalized reset target-clean. It queued current insertion remainders and added propagated old queue width only to output/range boxes, but the raw diagnostics did not expose full Phi_L entries and the scalar direction was still a limited approximation.

The new opt-in `symbolic_queue_mode="flowstar_linear_v2"` path, exposed by `reset_mode="normalized_insertion_symqueue_v2"`, stores interval columns `J`, accumulated linear maps `Phi_L`, and inverse normalized-insertion scale factors. It extracts the degree-1 part of the inserted endpoint map as the current linear map, propagates old interval columns through that map, and queues the current inserted endpoint remainder as a new source for future steps.

## Flow* Behavior Approximated

This approximates Flow*'s separation between ordinary local Taylor-model remainder validation and symbolic remainder propagation through linear/preconditioning maps. The v2 path keeps propagated symbolic queue width out of the ordinary target remainder check and adds it back to the reported output range. Diagnostics report J count, Phi_L count, linear-map entries/norms, inverse scalars, ordinary step remainder, reset width, right-map range width, target-check width, and output-only symbolic width.

## What Remains Missing

This is still not Flow* parity. The implementation is clean-room and only uses the degree-1 map available from the normalized insertion Taylor model. It does not copy Flow* GPL source, does not claim exact source-level queue ordering, and does not implement every Flow* preconditioning and Horner insertion detail. Nonlinear polynomial terms remain in the ordinary Taylor model/range accounting rather than becoming queue entries.

## Why Experimental

The v2 queue is conservative by construction when `output_range_includes_symbolic_contributions` is true, but it may add output width without reducing reset width. H10 reachability, sample containment, and one-step oracle results must be read together before drawing mechanism conclusions. Do not claim Flow* parity unless an h10 run is reached and the Flow* comparison supports that claim.
