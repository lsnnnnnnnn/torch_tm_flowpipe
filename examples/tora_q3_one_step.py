"""Portable one-leaf, one-step TORA-Q3 example."""

from __future__ import annotations

import torch

from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    dense_tora_q3_dr_step,
    tora_b48_boxes,
)


def main() -> None:
    state_lower, state_upper = tora_b48_boxes()
    control_lower = torch.tensor([9.8], dtype=torch.float64)
    control_upper = torch.tensor([10.2], dtype=torch.float64)
    initial = build_tora_q3_box_model(
        state_lower[:1], state_upper[:1], control_lower, control_upper
    )
    result = dense_tora_q3_dr_step(initial)
    print(
        f"status={result.status} accepted_leaves="
        f"{int(result.accepted_by_leaf.sum())}/1"
    )


if __name__ == "__main__":
    main()
