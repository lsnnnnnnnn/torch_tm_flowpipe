#!/usr/bin/env python3
"""New small GPU equivalence gate; no old checker or numerical run is replayed."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from private_outputs import EXPORTS, install
from run_quad_candidate import load_baseline, write_new


def cases(torch, ck):
    device = "cuda:0"
    x = (torch.arange(21, device=device, dtype=torch.float64).reshape(3, 7) - 9) / 8
    a = torch.stack((x - 0.0625, x + 0.125), -1)
    b = torch.stack((-x - 0.25, -x + 0.03125), -1)
    idx = torch.tensor([0, 2, 5, 1, 6, 3, 4], device=device, dtype=torch.int64)
    offsets = torch.tensor([0, 2, 2, 7], device=device, dtype=torch.int64)
    one = torch.tensor([0, 7], device=device, dtype=torch.int64)
    empty = torch.empty(0, device=device, dtype=torch.int64)
    empty_segments = torch.tensor([0, 0, 0], device=device, dtype=torch.int64)
    return [
        ("seg_mul_iv", ck.seg_mul_iv, (a, b, idx, idx.flip(0), offsets)),
        ("seg_mul_iv_empty_segments", ck.seg_mul_iv, (a, b, empty, empty, empty_segments)),
        ("seg_mul_pt", ck.seg_mul_pt, (x, -x, idx, idx.flip(0), offsets)),
        ("seg_dot_pt_iv_single_shared", ck.seg_dot_pt_iv, (x, a[0], one, idx)),
        ("seg_dot_pt_iv_batched", ck.seg_dot_pt_iv, (x, a, offsets, idx)),
        ("seg_dot_pt_iv_empty", ck.seg_dot_pt_iv, (x, a[0], empty_segments[:2], empty)),
        ("iv_mul", ck.iv_mul, (a, b)),
        ("iv_addsub_add", ck.iv_add, (a, b)),
        ("iv_addsub_sub", ck.iv_sub, (a, b)),
        ("iv_neg", ck.iv_neg, (a,)),
        ("iv_mul_point", ck.iv_mul_point, (a, x)),
        ("iv_sum", ck.iv_sum, (a,)),
        ("iv_sum_empty_inner", ck.iv_sum, (a[:, :0],)),
        ("iv_dot_point_iv_shared", ck.iv_dot_point_iv, (x, a[0])),
        ("iv_dot_point_iv_batched", ck.iv_dot_point_iv, (x, a)),
        ("iv_dot_point_iv_empty_inner", ck.iv_dot_point_iv, (x[:, :0], a[0, :0])),
    ]


def copy_args(torch, arguments, strided):
    copied = []
    for value in arguments:
        if strided and value.numel():
            target = torch.empty((*value.shape, 2), dtype=value.dtype, device=value.device)[..., 0]
            target.copy_(value)
        else:
            target = value.clone()
        copied.append(target)
    return tuple(copied)


def tensor_bytes(torch, value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def graph_call(torch, call, arguments):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call(*arguments)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call(*arguments)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or set(os.sched_getaffinity(0)) != {14, 15, 16, 17}:
        raise RuntimeError("requires physical GPU3 and CPUs14-17")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_new(output / "START.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new finite input GPU gate; not old 237-case checker", "exports": list(EXPORTS),
        "no_digest_operations": True, "no_jit": True, "gpu": "3",
        "cpu_affinity": sorted(os.sched_getaffinity(0))})
    result = {"status": "FIRST_REFUSAL", "passed_pairs": 0, "end_to_end_strict_certificate": False}
    binding = None
    current = None
    try:
        baseline = load_baseline()
        prepared = baseline.prepare(1)
        torch = prepared[0]
        from flowstar_gpu import cuda_kernels as ck, private_output_kernels
        # Both arms keep deterministic allocation filling enabled throughout.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        deterministic = (torch.are_deterministic_algorithms_enabled(),
                         torch.utils.deterministic.fill_uninitialized_memory)
        with (output / "CHECKS.jsonl").open("x") as log:
            for name, call, original in cases(torch, ck):
                for strided in (False, True):
                    a = copy_args(torch, original, strided)
                    b = copy_args(torch, original, strided)
                    base_graph, base_output = graph_call(torch, call, a)
                    binding = install(torch, ck, private_output_kernels)
                    private_graph, private_output = graph_call(torch, call, b)
                    receipt = binding.receipt
                    binding.restore()
                    binding = None
                    for shift in (0.0, 0.125, -0.375):
                        for aa, bb, source in zip(a, b, original):
                            value = source + shift if source.is_floating_point() else source
                            aa.copy_(value)
                            bb.copy_(value)
                        for mode in ("eager", "graph"):
                            current = {"case": name, "strided_inputs": strided, "shift": shift, "mode": mode}
                            if mode == "eager":
                                before = call(*a)
                                binding = install(torch, ck, private_output_kernels)
                                after = call(*b)
                                binding.restore()
                                binding = None
                            else:
                                base_graph.replay()
                                private_graph.replay()
                                before, after = base_output, private_output
                            torch.cuda.synchronize()
                            if before.shape != after.shape or tensor_bytes(torch, before) != tensor_bytes(torch, after):
                                raise RuntimeError("candidate differs from base output bytes")
                            if deterministic != (torch.are_deterministic_algorithms_enabled(),
                                                 torch.utils.deterministic.fill_uninitialized_memory):
                                raise RuntimeError("global deterministic settings changed")
                            row = dict(current, shape=list(after.shape), byte_equal=True,
                                       bytes=after.numel() * after.element_size())
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            result["passed_pairs"] += 1
                    del base_graph, private_graph, base_output, private_output
        result.update(status="PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE", exports=list(EXPORTS),
                      binding_receipt=receipt, deterministic_algorithms=True,
                      deterministic_fill=True, global_settings_unchanged=True,
                      empty_scope="empty input segments and inner reductions with nonzero output; zero-output kernel grids excluded",
                      scope="finite byte-equivalence checks only; no independent arithmetic or NNCS certificate")
        code = 0
    except BaseException as error:
        result.update(at=current, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        code = 1
    finally:
        if binding is not None:
            binding.restore()
        result.update(exit_code=code, elapsed_s=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "RESULT.json", result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
