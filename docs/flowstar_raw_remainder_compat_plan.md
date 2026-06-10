# Flow* Raw Remainder Compatibility Plan

This note freezes the current diagnostic state before adding the opt-in raw-remainder compatibility path.

## Why B-line is paused

The B-line NNCS/GPU work established downstream feasibility, but it is not the primary blocker for Flow* plant-core replication. Continuing NNCS or GPU demos now would optimize around a plant integrator mismatch that is already localized.

## Why return to the Flow* plant core

The current objective is to reproduce Flow* core Taylor-model plant behavior first, then accelerate it. The latest same-t/h diagnostics show the full-step validation candidate tube is width-close, the polynomial range is the same, cutoff/polyDiff is not the exposed cause, and the target remainder is the same.

## Known mismatch

At the first Van der Pol same-t/h divergence near t=0 with h=0.025, Flow* rejects because the raw Picard_ctrunc_normal y residual exceeds 1e-4. Current PyTorch target_remainder_flowstar_ctrunc accepts because its raw returned y remainder is below 1e-4. The internal Flow* audit localizes the exposed raw y_hi gap to accumulated remainder before the x0 add.

## Target one-step behavior

Add an experimental mode, `validation_mode="flowstar_raw_remainder_compat"`, that checks the replayed raw remainder accumulation used by the Flow* Expression path. The mode should reject the h=0.025 one-step candidate like Flow*, while recording whether residual_y_hi matches the Flow* probe within tolerance.

## Non-goals

- No NNCS work.
- No GPU demos.
- No h10 rerun before one-step compatibility is checked.
- No symbolic queue variants.
- No Flow* parity claim.
- No default solver behavior change.
