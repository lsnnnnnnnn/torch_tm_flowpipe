# Huan frozen Van der Pol Phase E — 2026-08-26

Primary status: `HUAN_PROOF_CONTRACT_CLOSED__VDP_CONTRACT_NOT_PORTABLE`

Phase E started only after the final scientific D1–D6 gate passed for Huan
`b0ff55745d69205f3afb4dc8077b9ac1310bfff3`. The package verifier mentioned
below checks artifacts only; it is not the scientific authorization gate.

## Frozen request

The requested polynomial system, initial box, complete O4 basis, fixed
`h=0.01`, ordinary remainder `[-1e-4,1e-4]` per component, cutoff `1e-10`,
validation threshold `1e-12`, native `h_min=0.002`, native `h_max=0.1`, and
symbolic-remainder queue 100 were held fixed. The detailed cross-lane mapping is
in `vdp/contract_matrix.csv`.

## Fresh Huan fixed-step results

All eight applicable CUDA runs completed. Endpoint and final-segment tube
channels remain separate.

| Mode | Horizon | Steps | Refinement iterations | Endpoint widths x / y | Final-segment tube widths x / y | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| parity | 0.01 | 1 | 4 | 0.3008966405 / 0.1213130194 | 0.3252453655 / 0.1494062018 | 1.012 |
| strict | 0.01 | 1 | 3 | 0.3008972926 / 0.1213154364 | 0.3252460176 / 0.1494086187 | 0.050 |
| parity | 1 | 100 | 438 | 0.0864639246 / 0.1128341235 | 0.0907291309 / 0.1270292613 | 2.811 |
| strict | 1 | 100 | 345 | 0.0866599385 / 0.1129654661 | 0.0909251799 / 0.1271676537 | 3.305 |
| parity | 3 | 300 | 1400 | 0.1727819871 / 0.1360494884 | 0.1982528193 / 0.1529438154 | 8.740 |
| strict | 3 | 300 | 1134 | 0.1738511946 / 0.1370129890 | 0.1993223687 / 0.1539077579 | 10.230 |
| parity | 6.32 | 632 | 3069 | 0.1895657269 / 0.1546415578 | 0.2148751692 / 0.1722329935 | 18.400 |
| strict | 6.32 | 632 | 2548 | 0.1945369351 / 0.1621766677 | 0.2198478847 / 0.1797713632 | 21.907 |

Runtime is diagnostic only: the first parity run includes CUDA warm-up, there
were no repetitions, and Phase F was excluded. No throughput conclusion follows.

At step 1 both initial self maps pass both components. The parity and strict
candidate hashes are respectively
`975dd4e1efefb1580556b8c144ecf937253d6efd9455fcd3ae2f051854c57f22`
and
`e283048acab48c75929cc796c101da63a314e39bb2690673746f16dad870346c`.
The complete proposals, component margins, sequential commits, final owners,
endpoint/tube boxes, and ordinary remainders are in the Huan run index and the
compressed refinement ledger.

## Mandatory native stop

Both Huan modes reject the exact native setting during `Settings`
construction:

```text
adaptive stepsize with symbolic remainders is not implemented
```

The exact frozen native lane requires both adaptive `0.002..0.1` and symbolic
queue 100. Disabling symbolic remainder, changing the step policy, or altering
another setting would violate the goal. Consequently:

- Huan parity native T=10: `NOT_RUN_CONTRACT_NOT_PORTABLE`.
- Huan strict native T=10: `NOT_RUN_CONTRACT_NOT_PORTABLE`.
- Fresh stock Flow* and Torch C2 ranking runs after discovery:
  `NOT_RUN_AFTER_CONTRACT_PORTABILITY_STOP`.
- Cross-tool first divergence: `NOT_ADJUDICATED_CONTRACT_PORTABILITY_STOP`.
- Huan explanation of Torch C2's terminal `y`-upper first-self-map failure:
  `NOT_ADJUDICATED`.

This is not a claim that parity or strict fails to reach T=10. It is the
stronger procedural statement that the requested native experiment cannot be
represented by the repaired engine without changing the scientific contract.
The fixed T=6.32 completion does not substitute for native T=10.

The required top-level Phase-E files include explicit empty scientific fields
for the stopped Flow*/Torch rows. The artifact verifier rejects any fabricated
endpoint, width, runtime, or step count in those rows.

## Authorization boundary

Throughput status is
`NOT_RUN_THIS_ROUND_AFTER_PROOF_AND_VDP_SCOPE`. No B=1…4096 performance campaign,
GPU speedup claim, four-tool winner table, or strict/parity T=10 cost claim is
authorized. A subsequent round needs a reviewed, tested adaptive+SR100 engine
implementation that leaves the frozen numerical settings unchanged, followed
by a fresh scientific gate and four-lane Phase E.
