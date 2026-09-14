"""Time the old CPU block and resident B32 block on captured real inputs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import gzip
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.resident_tm_block.exact_oracle import verify_result
from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.flowpipe import insert_ctrunc_normal_dependency_preserving
from torch_tm_flowpipe.resident_tm_block import (
    ResidentNormalCompositionRequest,
    ResidentTMBlockExecutor,
    canonical_exponents,
    request_from_taylor_models,
    resident_cuda_startup_check,
)


def _tensor(record: dict) -> torch.Tensor:
    assert record["tensor"] in {"torch.float64", "torch.int64"}
    dtype = torch.float64 if record["tensor"] == "torch.float64" else torch.int64
    values = (
        [float.fromhex(value) for value in record["values"]]
        if dtype == torch.float64
        else record["values"]
    )
    return torch.tensor(values, dtype=dtype).reshape(record["shape"])


def _float(record):
    return None if record is None else float.fromhex(record["float_hex"])


def decode_request(record: dict) -> ResidentNormalCompositionRequest:
    assert record["type"] == "ResidentNormalCompositionRequest"
    fields = record["fields"]
    tensor_names = (
        "outer_point", "outer_lo", "outer_hi", "outer_rem_lo", "outer_rem_hi",
        "inner_point", "inner_lo", "inner_hi", "inner_rem_lo", "inner_rem_hi",
        "domain_lo", "domain_hi",
    )
    values = {name: _tensor(fields[name]) for name in tensor_names}
    return ResidentNormalCompositionRequest(
        fields["request_id"], **values,
        outer_splits=tuple(fields["outer_splits"]),
        inner_splits=tuple(fields["inner_splits"]), order=int(fields["order"]),
        cutoff=_float(fields["cutoff"]), scalar_output=bool(fields["scalar_output"]),
        basis_fingerprint=fields["basis_fingerprint"],
    )


def captured_first_requests(path: Path) -> list[ResidentNormalCompositionRequest]:
    chosen: dict[str, ResidentNormalCompositionRequest] = {}
    with gzip.open(path / "lifecycle.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            request = event.get("request")
            if (
                event.get("event") == "submit"
                and request is not None
                and request.get("type") == "ResidentNormalCompositionRequest"
                and event["task"] not in chosen
            ):
                chosen[event["task"]] = decode_request(request)
    return [chosen[str(index)] for index in sorted(map(int, chosen))]


def request_models(request: ResidentNormalCompositionRequest) -> tuple[TMVector, TMVector, list[Interval]]:
    exponents = canonical_exponents(request.order)
    domain = [
        Interval(request.domain_lo[index], request.domain_hi[index]) for index in range(2)
    ]

    def models(point, rem_lo, rem_hi, splits):
        result = []
        for row in range(point.shape[0]):
            terms = {
                exponent: point[row, index].clone()
                for index, exponent in enumerate(exponents)
                if bool(point[row, index] != 0)
            }
            result.append(
                TaylorModel(
                    Polynomial(terms, 2), Interval(rem_lo[row], rem_hi[row]), domain,
                    order=request.order,
                    truncation_range_split=(splits[row] if splits[row] > 1 else None),
                )
            )
        return TMVector(result)

    return (
        models(
            request.outer_point, request.outer_rem_lo, request.outer_rem_hi,
            request.outer_splits,
        ),
        models(
            request.inner_point, request.inner_rem_lo, request.inner_rem_hi,
            request.inner_splits,
        ),
        domain,
    )


def _range_widths(value: TMVector) -> list[float]:
    return [float(interval.width()) for interval in value.range_box()]


def benchmark_one(plant: str, path: Path, repetitions: int) -> dict:
    captured = captured_first_requests(path)
    if len(captured) != 32:
        raise AssertionError(f"expected 32 distinct real requests, got {len(captured)}")
    triples = [request_models(request) for request in captured]
    executor = ResidentTMBlockExecutor()
    resident_cuda_startup_check()
    warm_requests = [
        request_from_taylor_models(
            f"warm-{index}", outer, inner, captured[index].order,
            captured[index].cutoff, domain,
        )
        for index, (outer, inner, domain) in enumerate(triples)
    ]
    assert all(request is not None for request in warm_requests)
    warm_results = executor.evaluate(warm_requests)
    for request in warm_requests:
        result = warm_results[request.request_id]
        assert result.ok
        verify_result(request, result)

    samples = []
    final_cpu = final_gpu = None
    orders = (("cpu", "gpu"), ("gpu", "cpu"), ("cpu", "gpu"), ("gpu", "cpu"), ("cpu", "gpu"))
    if repetitions != len(orders):
        raise ValueError("the frozen local benchmark uses exactly five pairs")
    for repetition, route_order in enumerate(orders):
        values = {}
        for route in route_order:
            begin = time.perf_counter_ns()
            thread_begin = time.thread_time_ns()
            if route == "cpu":
                output = [
                    insert_ctrunc_normal_dependency_preserving(
                        outer, inner, captured[index].order, captured[index].cutoff, domain
                    )
                    for index, (outer, inner, domain) in enumerate(triples)
                ]
                final_cpu = output
            else:
                requests = [
                    request_from_taylor_models(
                        f"rep{repetition}-{index}", outer, inner, captured[index].order,
                        captured[index].cutoff, domain,
                    )
                    for index, (outer, inner, domain) in enumerate(triples)
                ]
                assert all(request is not None for request in requests)
                results = executor.evaluate(requests)
                assert all(results[request.request_id].ok for request in requests)
                final_gpu = [results[request.request_id].output for request in requests]
            values[route] = dict(
                wall_s=(time.perf_counter_ns() - begin) / 1e9,
                thread_cpu_s=(time.thread_time_ns() - thread_begin) / 1e9,
            )
        samples.append(
            dict(
                repetition=repetition, order="/".join(route_order),
                cpu_wall_s=values["cpu"]["wall_s"], gpu_wall_s=values["gpu"]["wall_s"],
                cpu_thread_s=values["cpu"]["thread_cpu_s"],
                gpu_thread_s=values["gpu"]["thread_cpu_s"],
                cpu_over_gpu=values["cpu"]["wall_s"] / values["gpu"]["wall_s"],
            )
        )
    assert final_cpu is not None and final_gpu is not None
    ratios = []
    for cpu, gpu in zip(final_cpu, final_gpu):
        assert isinstance(cpu, TMVector) and isinstance(gpu, TMVector)
        for cpu_width, gpu_width in zip(_range_widths(cpu), _range_widths(gpu)):
            if cpu_width > 1e-15:
                ratios.append(gpu_width / cpu_width)
    speeds = [sample["cpu_over_gpu"] for sample in samples]
    return dict(
        plant=plant, captured_run=str(path), batch=32,
        captured_inputs_are_first_real_accepted_boundary_calls=True,
        repeated_saved_answers_used_to_advance=False,
        current_gpu_request_build_inside_timing=True,
        cpu_input_tm_already_available_as_at_production_call=True,
        gpu_timing_includes_pack_h2d_kernel_d2h_receipt_checks_and_object_rebuild=True,
        cold_compile_and_startup_outside_timing=True,
        pairs=samples, median_cpu_over_gpu=statistics.median(speeds),
        wins=sum(value > 1 for value in speeds), target_3x_met=statistics.median(speeds) >= 3,
        final_output_width_ratio_gpu_over_legacy_cpu_max=max(ratios, default=None),
        legacy_cpu_known_to_omit_retained_coefficient_roundoff=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = []
    for plant in ("van_der_pol", "brusselator"):
        results.append(
            benchmark_one(
                plant,
                args.diagnostic / f"prefix-b32-{plant}-Gr",
                args.repetitions,
            )
        )
    output = dict(
        schema="resident-tm-block-local-real-b32-timing-v1", passed=True,
        measurements=results,
        local_target_met_both=all(row["target_3x_met"] for row in results),
        timing_semantics={
            "wall": "one logical CPU core; each pair alternates route order",
            "cpu": "32 legacy production block calls on captured current inputs",
            "gpu": "32 request builds plus one resident group and complete returned objects",
        },
    )
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
