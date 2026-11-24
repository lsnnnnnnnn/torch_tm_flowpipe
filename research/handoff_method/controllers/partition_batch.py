"""DP v2: vectorized construction and 64-cell batching, original directed body.

There is no reduction over the enlarged batch dimension. Partition boundaries,
per-row arithmetic, whole-box diagnostics, and lexicographic hull order remain
unchanged. No model, precision, strict injection, or reachability step is altered.
"""
from itertools import product
from pathlib import Path
import time
from types import SimpleNamespace


CELLS = tuple(product(range(4), repeat=4))


def partition_chunks(torch, iv, hull, counts=None):
    """Gather original RN boundaries; certify the complete Cartesian cell list."""
    cells = CELLS
    if (len(cells) != 256 or any(cell != ((i // 64) % 4, (i // 16) % 4,
                                         (i // 4) % 4, i % 4)
                                 for i, cell in enumerate(cells))):
        raise RuntimeError("DP v2 requires exactly the original lexicographic split4 partition")
    low, high = hull[..., 0], hull[..., 1]
    boundaries = [low] + [low + (high - low) * (cut / 4)
                          for cut in range(1, 4)] + [high]
    if not all(bool((torch.isfinite(boundaries[cut])
                     & (boundaries[cut - 1] <= boundaries[cut])).all())
               for cut in range(1, 5)):
        raise FloatingPointError("DP controller boundaries failed to partition input box")
    boundary_values = torch.stack(boundaries, dim=0)
    indices = torch.tensor(cells, dtype=torch.int64, device=hull.device)
    lanes = torch.arange(hull.shape[0], device=hull.device)[None, :, None]
    axes = torch.arange(4, device=hull.device)[None, None, :]
    first_low = last_high = None
    for start in range(0, 256, 64):
        selected = indices[start:start + 64, None, :]
        lower = boundary_values[selected, lanes, axes]
        upper = boundary_values[selected + 1, lanes, axes]
        chunk = torch.stack((lower, upper), dim=-1)
        # This all-element predicate is the conjunction of the same 64 checks.
        # No affine operation or accepted result is performed with invalid cells.
        iv.assert_valid(chunk, "DP v2 controller partition batch")
        if first_low is None:
            first_low = chunk[0, ..., 0]
        last_high = chunk[-1, ..., 1]
        if counts is not None:
            counts["partition_batch_assertions"] += 1
        yield chunk
    # Complete Cartesian metadata + monotone boundaries proves coverage. These
    # two endpoint equalities additionally verify its actual gather realization.
    if not (torch.equal(first_low, low) and torch.equal(last_high, high)):
        raise AssertionError("DP v2 gathered partition does not cover input box")


def evaluate(torch, iv, affine, hull, T, layers, *, parts=4, counts=None):
    if parts != 4:
        raise ValueError("DP v2 is qualified only for split4")
    B = hull.shape[0]
    if (hull.ndim != 3 or hull.shape[1:] != (4, 2) or not 1 <= B <= 225
            or T.shape != (B, 2, 4) or hull.dtype != torch.float64
            or T.dtype != torch.float64 or T.device != hull.device):
        raise ValueError("DP batching requires binary64 hull[B,4,2], T[B,2,4], B<=225")
    network, whole, layers_diagnostics = affine(hull, T, layers)
    if counts is not None:
        counts["whole_calls"] += 1
    joined = None
    for chunk in partition_chunks(torch, iv, hull, counts):
        C = chunk.shape[0]
        batched_hull = chunk.reshape(C * B, 4, 2)
        batched_T = T.unsqueeze(0).expand(C, -1, -1, -1).reshape(C * B, 2, 4).contiguous()
        _, residuals, _ = affine(batched_hull, batched_T, layers)
        # Keep the original cell order, including signed-zero tie semantics.
        for sub in residuals.reshape(C, B, 2, 2).unbind(0):
            joined = sub.clone() if joined is None else torch.stack(
                (torch.minimum(joined[..., 0], sub[..., 0]),
                 torch.maximum(joined[..., 1], sub[..., 1])), dim=-1)
        if counts is not None:
            counts["batched_subbox_calls"] += 1
            counts["subboxes_evaluated"] += C
            counts["subbox_lane_rows"] += C * B
            counts["peak_batch_rows"] = max(counts["peak_batch_rows"], C * B)
    iv.assert_valid(joined, "DP controller partition union")
    tight = torch.stack((torch.maximum(whole[..., 0], joined[..., 0]),
                         torch.minimum(whole[..., 1], joined[..., 1])), dim=-1)
    iv.assert_valid(tight, "DP controller residual intersection")
    whole_width, tight_width = whole[..., 1] - whole[..., 0], tight[..., 1] - tight[..., 0]
    diagnostics = {
        "whole_layers": layers_diagnostics,
        "subboxes_per_input_box": parts ** 4,
        "global_T_reused": True,
        "whole_width_max": float(whole_width.max().item()),
        "union_width_max": float((joined[..., 1] - joined[..., 0]).max().item()),
        "tight_width_max": float(tight_width.max().item()),
        "improved_outputs": int((tight_width < whole_width).sum().item()),
    }
    return network, tight, diagnostics


def assert_equal(torch, candidate, reference):
    for index in (0, 1):
        a, b = candidate[index], reference[index]
        if (a.shape != b.shape or a.dtype != b.dtype
                or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))):
            raise AssertionError(f"DP batching changed returned tensor {index}")
    if candidate[2] != reference[2]:
        raise AssertionError("DP batching changed public diagnostics")


def install(torch, module, iv):
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError("DP production batching requires the preloaded Torch 2.5.1 runtime")
    original = module.partitioned_affine_residual_interval
    affine = module.affine_residual_interval
    if (original.__name__ != "partitioned_affine_residual_interval"
            or original.__globals__ is not vars(module)
            or affine.__globals__ is not vars(module)):
        raise RuntimeError("DP batching requires the unwrapped controller functions")
    active = armed = restored = False
    references = []
    counts = dict.fromkeys(("calls", "completed", "failed", "whole_calls", "batched_subbox_calls",
                           "subboxes_evaluated", "subbox_lane_rows", "peak_batch_rows",
                           "live_reference_checks", "partition_batch_assertions"), 0)

    def candidate(hull, T, layers, parts=2):
        nonlocal active
        if active or restored:
            raise RuntimeError("DP batching binding is inactive or reentered")
        active = True
        counts["calls"] += 1
        try:
            result = evaluate(torch, iv, affine, hull, T, layers, parts=parts, counts=counts)
            if armed and not references:
                if hull.is_cuda:
                    torch.cuda.synchronize(hull.device)
                reference_started = time.perf_counter()
                reference = original(hull, T, layers, parts=parts)
                if hull.is_cuda:
                    torch.cuda.synchronize(hull.device)
                reference_wall_s = time.perf_counter() - reference_started
                assert_equal(torch, result, reference)
                references.append({"batch": hull.shape[0], "parts": parts,
                                   "subboxes_per_input_box": parts ** 4,
                                   "reference_wall_s": reference_wall_s,
                                   "reference_timing_scope": "original same-input reference call, synchronized before/after; excludes byte comparison; not subtracted from headline timing",
                                   "returned_tensor_bytes_equal": True,
                                   "all_public_diagnostics_equal": True})
                counts["live_reference_checks"] += 1
            counts["completed"] += 1
            return result
        except Exception:
            counts["failed"] += 1
            raise
        finally:
            active = False

    def arm_live_reference():
        nonlocal armed
        if active or armed or restored:
            raise RuntimeError("DP live reference cannot be armed in this state")
        armed = True

    def restore():
        nonlocal restored
        if restored:
            return
        if active or module.partitioned_affine_residual_interval is not candidate:
            raise RuntimeError("DP batching restore ownership mismatch")
        module.partitioned_affine_residual_interval = original
        restored = True

    module.partitioned_affine_residual_interval = candidate
    return SimpleNamespace(original=original, candidate=candidate, counters=counts,
                           reference_records=references, arm_live_reference=arm_live_reference,
                           restore=restore, receipt={
        "policy": "dp_v2_vectorized64_cells_original_directed_body_ordered_hull",
        "source": str(Path(__file__).resolve()), "controller_source": module.__file__,
        "implementation_version": 2,
        "cell_chunk": 64, "maximum_original_batch": 225, "maximum_enlarged_batch": 14400,
        "partition_construction": "gather unchanged RN boundaries by verified lexicographic Cartesian metadata",
        "partition_validity": "same all-element finite/order predicate once per 64-cell chunk",
        "coverage": "complete Cartesian enumeration, monotone boundaries, actual extreme-corner equality",
        "original_affine_body_unchanged": True, "partition_count_unchanged": True,
        "lexicographic_minmax_order_retained": True, "content_digest_calls": 0,
        "jit_or_extension_builds": 0,
    })
