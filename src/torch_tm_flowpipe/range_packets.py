"""Bounded heterogeneous CUDA work packets for existing range requests.

This opt-in path changes physical organization only.  Every request keeps its
own support, output count, variable roles, power entries, term-operation chain,
ordered sum, identity and private result.  Numerical values are copied into one
owned contiguous host payload, then transferred with one metadata and one
numeric H2D operation.  A service-owned executor reuses bounded pinned/device
storage only after the preceding packet has completed and its results were
copied into private CPU tensors.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Sequence

import torch

from .packed_boundary_range import make_plan
from .range_cuda import FLAGS, _check
from .range_requests import (
    RangeRequest, RangeResult, _validate_structure, evaluate_range_requests,
)


PACKET_MAGIC = 0x5257504B543031
PACKET_VERSION = 1
HEADER_WIDTH = 24
REQUEST_WIDTH = 18
POWER_WIDTH = 3
CELL_WIDTH = 5
OUTPUT_WIDTH = 4
VARIABLE_WIDTH = 3

(H_MAGIC, H_VERSION, H_BUFFER_EPOCH, H_GLOBAL_STATUS,
 H_RECEIPT_VALIDATE, H_RECEIPT_POWER, H_RECEIPT_TERM, H_RECEIPT_SUM,
 H_REQUESTS, H_POWERS, H_CELLS, H_OUTPUTS, H_VARIABLES, H_OPERATIONS,
 H_NUMERIC_LENGTH, H_OUTPUT_LENGTH, H_STATUS_OFFSET, H_REQUEST_OFFSET,
 H_POWER_OFFSET, H_CELL_OFFSET, H_OUTPUT_OFFSET, H_VARIABLE_OFFSET,
 H_OPERATION_OFFSET, H_METADATA_LENGTH) = range(HEADER_WIDTH)

(R_TOKEN, R_COEFF_LO, R_COEFF_HI, R_DOMAIN_LO, R_DOMAIN_HI,
 R_OUTPUT_START, R_OUTPUT_COUNT, R_CELL_START, R_CELL_COUNT,
 R_POWER_START, R_POWER_COUNT, R_VARIABLE_START, R_VARIABLE_COUNT,
 R_OPERATION_START, R_OPERATION_COUNT, R_TERMS,
 R_POINT_COEFFICIENTS, R_ENABLED) = range(REQUEST_WIDTH)


class PacketCapacityError(ValueError):
    """One request or packet exceeds a preregistered resource bound."""


class PacketStructureError(RuntimeError):
    """A packet descriptor/envelope was corrupted and produced no certificate."""


@dataclass(frozen=True)
class PacketLimits:
    """Fixed safety limits; formal experiments must record these exact values."""

    max_requests: int = 32
    max_terms: int = 262_144
    max_coefficients: int = 262_144
    max_variables: int = 2_048
    max_powers: int = 131_072
    max_operations: int = 1_048_576
    max_outputs: int = 2_048
    max_metadata_bytes: int = 64 * 1024 * 1024
    max_numeric_bytes: int = 64 * 1024 * 1024
    max_output_bytes: int = 128 * 1024 * 1024
    max_packet_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        launch_limit = 2**31 - 1
        for name in ("max_requests", "max_terms", "max_coefficients", "max_variables",
                     "max_powers", "max_operations", "max_outputs"):
            if getattr(self, name) > launch_limit:
                raise ValueError(f"{name} exceeds the CUDA launch descriptor range")
        for name in ("max_metadata_bytes", "max_numeric_bytes", "max_output_bytes"):
            if getattr(self, name) > launch_limit * 8:
                raise ValueError(f"{name} exceeds the CUDA signed-int item range")


DEFAULT_PACKET_LIMITS = PacketLimits()


@dataclass(frozen=True)
class PacketRequirements:
    requests: int = 0
    terms: int = 0
    coefficients: int = 0
    variables: int = 0
    powers: int = 0
    operations: int = 0
    outputs: int = 0
    metadata_items: int = HEADER_WIDTH
    numeric_items: int = 0
    output_items: int = 0

    @property
    def metadata_bytes(self):
        return self.metadata_items * 8

    @property
    def numeric_bytes(self):
        return self.numeric_items * 8

    @property
    def output_bytes(self):
        return self.output_items * 8

    @property
    def packet_bytes(self):
        return self.metadata_bytes + self.numeric_bytes + self.output_bytes


@dataclass(frozen=True)
class _RequestLayout:
    powers: tuple[tuple[int, int], ...]
    term_operations: tuple[tuple[int, ...], ...]


@lru_cache(maxsize=512)
def _request_layout(exponents, variables, kind, state_variables, time_variable):
    plan = make_plan(exponents, variables, state_variables if kind == "normal" else None,
                     time_variable)
    powers = tuple(sorted({(variable, power)
                           for stage, variable, _, values in plan.stages
                           if stage == "power" for power in values}))
    lookup = {key: index for index, key in enumerate(powers)}
    operations = [[] for _ in exponents]
    constants = {-1.0: -2, 0.0: -3, 1.0: -4}
    for stage, variable, indices, values in plan.stages:
        for term, value in zip(indices, values):
            operations[term].append(lookup[variable, value] if stage == "power"
                                    else constants[value])
    return _RequestLayout(powers, tuple(tuple(row) for row in operations))


def _shape(request):
    outputs, terms = request.coefficients_lo.shape
    variables = request.domain_lo.numel()
    layout = _request_layout(request.exponents, variables, request.kind,
                             request.state_variables, request.time_variable)
    operations = sum(map(len, layout.term_operations))
    return outputs, terms, variables, layout, operations


def _requirements(requests):
    rows = [_shape(request) for request in requests]
    request_count = len(rows)
    terms = sum(row[1] for row in rows)
    coefficients = sum(row[0] * row[1] for row in rows)
    variables = sum(row[2] for row in rows)
    powers = sum(len(row[3].powers) for row in rows)
    operations = sum(row[4] for row in rows)
    outputs = sum(row[0] for row in rows)
    return _requirements_from_counts(request_count, terms, coefficients, variables,
                                     powers, operations, outputs)


def _requirements_from_counts(request_count, terms, coefficients, variables,
                              powers, operations, outputs):
    metadata_items = (HEADER_WIDTH + request_count + request_count * REQUEST_WIDTH
                      + powers * POWER_WIDTH + coefficients * CELL_WIDTH
                      + outputs * OUTPUT_WIDTH + variables * VARIABLE_WIDTH
                      + operations)
    numeric_items = 2 * (coefficients + variables)
    output_items = 2 * (outputs + coefficients + powers)
    return PacketRequirements(request_count, terms, coefficients, variables, powers,
                              operations, outputs, metadata_items, numeric_items,
                              output_items)


def _combine_requirements(left, right):
    return _requirements_from_counts(
        left.requests + right.requests, left.terms + right.terms,
        left.coefficients + right.coefficients, left.variables + right.variables,
        left.powers + right.powers, left.operations + right.operations,
        left.outputs + right.outputs)


def _fits(requirements, limits):
    pairs = (
        (requirements.requests, limits.max_requests),
        (requirements.terms, limits.max_terms),
        (requirements.coefficients, limits.max_coefficients),
        (requirements.variables, limits.max_variables),
        (requirements.powers, limits.max_powers),
        (requirements.operations, limits.max_operations),
        (requirements.outputs, limits.max_outputs),
        (requirements.metadata_bytes, limits.max_metadata_bytes),
        (requirements.numeric_bytes, limits.max_numeric_bytes),
        (requirements.output_bytes, limits.max_output_bytes),
        (requirements.packet_bytes, limits.max_packet_bytes),
    )
    return all(value <= maximum for value, maximum in pairs)


def _capacity_message(requirements, limits):
    values = requirements.__dict__ | {
        "metadata_bytes": requirements.metadata_bytes,
        "numeric_bytes": requirements.numeric_bytes,
        "output_bytes": requirements.output_bytes,
        "packet_bytes": requirements.packet_bytes,
    }
    exceeded = []
    for name, maximum in (
        ("requests", limits.max_requests), ("terms", limits.max_terms),
        ("coefficients", limits.max_coefficients), ("variables", limits.max_variables),
        ("powers", limits.max_powers), ("operations", limits.max_operations),
        ("outputs", limits.max_outputs), ("metadata_bytes", limits.max_metadata_bytes),
        ("numeric_bytes", limits.max_numeric_bytes), ("output_bytes", limits.max_output_bytes),
        ("packet_bytes", limits.max_packet_bytes),
    ):
        if values[name] > maximum:
            exceeded.append(f"{name}={values[name]}>{maximum}")
    return "range packet capacity exceeded: " + ", ".join(exceeded)


def _request_token(request_id):
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class RangeWorkPacket:
    """Owned CPU payload plus checked offsets; no mutable caller tensor aliases."""

    requests: tuple[RangeRequest, ...]
    request_ids: tuple[str, ...]
    metadata: torch.Tensor
    numeric: torch.Tensor
    requirements: PacketRequirements


def build_range_work_packet(requests: Sequence[RangeRequest], *, limits=DEFAULT_PACKET_LIMITS):
    requests = tuple(requests)
    if not requests:
        raise ValueError("a range work packet must contain at least one request")
    ids = [request.request_id for request in requests]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique request IDs required")
    for request in requests:
        _validate_structure(request)
        if request.step_powers is not None:
            raise ValueError("external power tables remain on the original CPU fallback")
        if request.cancelled or not request.enabled:
            raise ValueError("cancelled or masked requests do not enter a device packet")
    requirements = _requirements(requests)
    if not _fits(requirements, limits):
        raise PacketCapacityError(_capacity_message(requirements, limits))

    layouts = [_shape(request) for request in requests]
    coefficients = requirements.coefficients
    variables = requirements.variables
    numeric_parts = []
    for field in ("coefficients_lo", "coefficients_hi", "domain_lo", "domain_hi"):
        numeric_parts.extend(getattr(request, field).detach().reshape(-1) for request in requests)
    numeric = (torch.cat(numeric_parts) if numeric_parts
               else torch.empty(0, dtype=torch.float64))
    assert numeric.numel() == requirements.numeric_items
    metadata_values = [0] * requirements.metadata_items

    status_offset = HEADER_WIDTH
    request_offset = status_offset + requirements.requests
    power_offset = request_offset + requirements.requests * REQUEST_WIDTH
    cell_offset = power_offset + requirements.powers * POWER_WIDTH
    output_offset = cell_offset + requirements.coefficients * CELL_WIDTH
    variable_offset = output_offset + requirements.outputs * OUTPUT_WIDTH
    operation_offset = variable_offset + requirements.variables * VARIABLE_WIDTH
    assert operation_offset + requirements.operations == requirements.metadata_items
    header = [
        PACKET_MAGIC, PACKET_VERSION, 0, 0, 0, 0, 0, 0,
        requirements.requests, requirements.powers, requirements.coefficients,
        requirements.outputs, requirements.variables, requirements.operations,
        requirements.numeric_items, requirements.output_items, status_offset,
        request_offset, power_offset, cell_offset, output_offset, variable_offset,
        operation_offset, requirements.metadata_items,
    ]
    metadata_values[:HEADER_WIDTH] = header

    coefficient_cursor = domain_cursor = output_cursor = cell_cursor = 0
    power_cursor = variable_cursor = operation_cursor = 0
    for request_index, (request, shaped) in enumerate(zip(requests, layouts)):
        outputs, terms, variable_count, layout, _ = shaped
        cells = outputs * terms
        cl = coefficient_cursor
        ch = coefficients + coefficient_cursor
        dl = 2 * coefficients + domain_cursor
        dh = 2 * coefficients + variables + domain_cursor
        descriptor = [
            _request_token(request.request_id), cl, ch, dl, dh,
            output_cursor, outputs, cell_cursor, cells,
            power_cursor, len(layout.powers), variable_cursor, variable_count,
            operation_cursor, shaped[4], terms,
            int(request.kind != "interval-coefficient"), 1,
        ]
        begin = request_offset + request_index * REQUEST_WIDTH
        metadata_values[begin:begin+REQUEST_WIDTH] = descriptor

        for local_variable in range(variable_count):
            begin = variable_offset + (variable_cursor + local_variable) * VARIABLE_WIDTH
            metadata_values[begin:begin+VARIABLE_WIDTH] = [
                request_index, local_variable, int(local_variable in request.state_variables)]
        for local_power, (variable, exponent) in enumerate(layout.powers):
            begin = power_offset + (power_cursor + local_power) * POWER_WIDTH
            metadata_values[begin:begin+POWER_WIDTH] = [request_index, variable, exponent]

        term_operation_starts = []
        for operations in layout.term_operations:
            term_operation_starts.append(operation_cursor)
            converted = [power_cursor + value if value >= 0 else value for value in operations]
            if converted:
                begin = operation_offset + operation_cursor
                metadata_values[begin:begin+len(converted)] = converted
            operation_cursor += len(converted)
        for output in range(outputs):
            begin = output_offset + (output_cursor + output) * OUTPUT_WIDTH
            metadata_values[begin:begin+OUTPUT_WIDTH] = [
                request_index, output, cell_cursor + output * terms, terms]
            for term in range(terms):
                cell = cell_cursor + output * terms + term
                begin = cell_offset + cell * CELL_WIDTH
                metadata_values[begin:begin+CELL_WIDTH] = [
                    request_index, output, term, term_operation_starts[term],
                    len(layout.term_operations[term])]

        coefficient_cursor += cells
        domain_cursor += variable_count
        output_cursor += outputs
        cell_cursor += cells
        power_cursor += len(layout.powers)
        variable_cursor += variable_count

    assert (coefficient_cursor, domain_cursor, output_cursor, cell_cursor,
            power_cursor, variable_cursor, operation_cursor) == (
                requirements.coefficients, requirements.variables, requirements.outputs,
                requirements.coefficients, requirements.powers, requirements.variables,
                requirements.operations)
    metadata = torch.tensor(metadata_values, dtype=torch.int64)
    return RangeWorkPacket(requests, tuple(ids), metadata, numeric, requirements)


def split_range_work_packets(requests: Sequence[RangeRequest], *, limits=DEFAULT_PACKET_LIMITS):
    """Greedily split at request boundaries; never split/reorder one request."""
    packets, current, oversize = [], [], []
    current_requirements = _requirements_from_counts(0, 0, 0, 0, 0, 0, 0)
    for request in requests:
        single = _requirements([request])
        requirements = _combine_requirements(current_requirements, single)
        if _fits(requirements, limits):
            current.append(request)
            current_requirements = requirements
            continue
        if current:
            packets.append(build_range_work_packet(current, limits=limits))
            current = []
            current_requirements = _requirements_from_counts(0, 0, 0, 0, 0, 0, 0)
        if _fits(single, limits):
            current = [request]
            current_requirements = single
        else:
            oversize.append((request, _capacity_message(single, limits)))
    if current:
        packets.append(build_range_work_packet(current, limits=limits))
    return packets, oversize


class PacketKernelModule:
    def __init__(self, device):
        start = time.perf_counter()
        self.device = device
        torch.cuda.init()
        torch.empty(1, device=device)
        libraries = sorted((Path(torch.__file__).resolve().parent.parent /
                            "nvidia/cuda_nvrtc/lib").glob("libnvrtc.so*"))
        if not libraries:
            raise RuntimeError("installed PyTorch NVRTC library not found")
        self.nvrtc = C.CDLL(str(libraries[0]))
        self.driver = C.CDLL("libcuda.so.1")
        self.nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p,
            C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
        self.nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
        for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
        self.driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p,
                                               C.POINTER(C.c_void_p), C.c_void_p]
        self.source = Path(__file__).with_name("range_packet_cuda_kernel.cu").read_bytes()
        program = C.c_void_p()
        _check(self.nvrtc.nvrtcCreateProgram(C.byref(program), self.source,
            b"range_packet_cuda_kernel.cu", 0, None, None), "nvrtcCreateProgram")
        try:
            options = (C.c_char_p * len(FLAGS))(*(flag.encode() for flag in FLAGS))
            code = self.nvrtc.nvrtcCompileProgram(program, len(FLAGS), options)
            size = C.c_size_t()
            _check(self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)), "nvrtcGetProgramLogSize")
            log = C.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            self.log = log.value.decode()
            if code:
                raise RuntimeError(f"NVRTC packet compilation failed ({code}): {self.log}")
            _check(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)), "nvrtcGetPTXSize")
            ptx = C.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module = C.c_void_p()
        _check(self.driver.cuModuleLoadData(C.byref(self.module), ptx), "cuModuleLoadData")
        self.functions = {}
        for name in ("packet_validate", "packet_powers", "packet_terms", "packet_sum",
                     "packet_arithmetic_probe"):
            function = C.c_void_p()
            _check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()),
                   "cuModuleGetFunction")
            self.functions[name] = function
        major, minor = C.c_int(), C.c_int()
        _check(self.nvrtc.nvrtcVersion(C.byref(major), C.byref(minor)), "nvrtcVersion")
        self.build = dict(nvrtc_path=str(libraries[0]), nvrtc_version=[major.value, minor.value],
            flags=list(FLAGS), source_sha256=hashlib.sha256(self.source).hexdigest(),
            ptx_sha256=hashlib.sha256(ptx.raw).hexdigest(),
            compile_and_load_s=time.perf_counter()-start, compiler_log=self.log,
            device=str(device), device_name=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)))

    def launch(self, name, count, args):
        values = [C.c_void_p(value.data_ptr()) if isinstance(value, torch.Tensor)
                  else C.c_int(value) for value in args]
        pointers = (C.c_void_p * len(values))(
            *(C.cast(C.byref(value), C.c_void_p) for value in values))
        _check(self.driver.cuLaunchKernel(self.functions[name], max(1, (count+127)//128),
            1, 1, 128, 1, 1, 0,
            C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream), pointers, None), name)


_MODULES = {}
_MODULE_LOCK = threading.Lock()


def get_packet_module(device="cuda:0"):
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device), _MODULE_LOCK:
        if device not in _MODULES:
            _MODULES[device] = PacketKernelModule(device)
        return _MODULES[device]


class _Scratch:
    """One synchronous executor owns these buffers until private scatter ends."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.host, self.gpu, self.capacity = {}, {}, {}
        self.device_allocation_operations = 0
        self.pinned_allocation_operations = 0
        self.cumulative_device_allocation_bytes = 0
        self.cumulative_pinned_allocation_bytes = 0

    @staticmethod
    def _next_capacity(count):
        return max(1, 1 << max(0, count-1).bit_length())

    def ensure(self, name, count, dtype):
        if name in self.capacity and self.capacity[name] >= count:
            return False
        capacity = self._next_capacity(count)
        item_size = torch.empty((), dtype=dtype).element_size()
        self.host[name] = torch.empty(capacity, dtype=dtype, pin_memory=True)
        self.gpu[name] = torch.empty(capacity, dtype=dtype, device=self.device)
        self.capacity[name] = capacity
        self.pinned_allocation_operations += 1
        self.device_allocation_operations += 1
        self.cumulative_pinned_allocation_bytes += capacity * item_size
        self.cumulative_device_allocation_bytes += capacity * item_size
        return True

    def snapshot(self):
        current_bytes = sum(self.capacity[name] * self.host[name].element_size()
                            for name in self.capacity)
        return dict(capacity=dict(self.capacity),
            device_allocation_operations=self.device_allocation_operations,
            pinned_allocation_operations=self.pinned_allocation_operations,
            cumulative_device_allocation_bytes=self.cumulative_device_allocation_bytes,
            cumulative_pinned_allocation_bytes=self.cumulative_pinned_allocation_bytes,
            current_device_capacity_bytes=current_bytes,
            current_pinned_capacity_bytes=current_bytes)


class RangePacketExecutor:
    """Serialize packets on the caller's CUDA stream with bounded scratch reuse."""

    def __init__(self, *, device="cuda:0", limits=DEFAULT_PACKET_LIMITS):
        self.device = torch.device(device)
        self.limits = limits
        self.module = get_packet_module(self.device)
        self.scratch = _Scratch(self.device)
        self._lock = threading.Lock()
        self._buffer_epoch = 0
        with torch.cuda.device(self.device):
            self._events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(7))

    def _check_envelope(self, packet):
        r = packet.requirements
        if not _fits(r, self.limits):
            raise PacketCapacityError(_capacity_message(r, self.limits))
        if packet.metadata.dtype != torch.int64 or packet.metadata.device.type != "cpu" or not packet.metadata.is_contiguous():
            raise PacketStructureError("packet metadata must be contiguous CPU int64")
        if packet.numeric.dtype != torch.float64 or packet.numeric.device.type != "cpu" or not packet.numeric.is_contiguous():
            raise PacketStructureError("packet numeric payload must be contiguous CPU binary64")
        if packet.metadata.numel() != r.metadata_items or packet.numeric.numel() != r.numeric_items:
            raise PacketStructureError("packet payload length differs from its owned envelope")
        if len(packet.requests) != r.requests or packet.request_ids != tuple(row.request_id for row in packet.requests):
            raise PacketStructureError("packet request identity envelope changed")
        header = packet.metadata[:HEADER_WIDTH].tolist()
        expected = (PACKET_MAGIC, PACKET_VERSION, r.requests, r.powers, r.coefficients,
                    r.outputs, r.variables, r.operations, r.numeric_items, r.output_items,
                    r.metadata_items)
        actual = (header[H_MAGIC], header[H_VERSION], header[H_REQUESTS], header[H_POWERS],
                  header[H_CELLS], header[H_OUTPUTS], header[H_VARIABLES],
                  header[H_OPERATIONS], header[H_NUMERIC_LENGTH], header[H_OUTPUT_LENGTH],
                  header[H_METADATA_LENGTH])
        if actual != expected:
            raise PacketStructureError("packet header differs from declared requirements")
        expected_offsets = (
            HEADER_WIDTH,
            HEADER_WIDTH + r.requests,
            HEADER_WIDTH + r.requests + r.requests*REQUEST_WIDTH,
            HEADER_WIDTH + r.requests + r.requests*REQUEST_WIDTH + r.powers*POWER_WIDTH,
            HEADER_WIDTH + r.requests + r.requests*REQUEST_WIDTH + r.powers*POWER_WIDTH
                + r.coefficients*CELL_WIDTH,
            HEADER_WIDTH + r.requests + r.requests*REQUEST_WIDTH + r.powers*POWER_WIDTH
                + r.coefficients*CELL_WIDTH + r.outputs*OUTPUT_WIDTH,
            HEADER_WIDTH + r.requests + r.requests*REQUEST_WIDTH + r.powers*POWER_WIDTH
                + r.coefficients*CELL_WIDTH + r.outputs*OUTPUT_WIDTH
                + r.variables*VARIABLE_WIDTH,
        )
        actual_offsets = tuple(header[index] for index in (
            H_STATUS_OFFSET, H_REQUEST_OFFSET, H_POWER_OFFSET, H_CELL_OFFSET,
            H_OUTPUT_OFFSET, H_VARIABLE_OFFSET, H_OPERATION_OFFSET))
        if actual_offsets != expected_offsets:
            raise PacketStructureError("packet section offsets differ from bounded layout")
        if any(header[index] for index in range(H_BUFFER_EPOCH, H_RECEIPT_SUM+1)):
            raise PacketStructureError("packet status and receipts were not freshly initialized")
        if bool(torch.any(packet.metadata[header[H_STATUS_OFFSET]:header[H_REQUEST_OFFSET]] != 0)):
            raise PacketStructureError("packet request status was not freshly initialized")
        request_offset = header[H_REQUEST_OFFSET]
        for index, request_id in enumerate(packet.request_ids):
            if int(packet.metadata[request_offset + index*REQUEST_WIDTH + R_TOKEN]) != _request_token(request_id):
                raise PacketStructureError("packet request token differs from request identity")

    def execute_packet(self, packet: RangeWorkPacket, *, diagnostics=False, timings=None):
        envelope_start = time.perf_counter()
        self._check_envelope(packet)
        envelope_check_s = time.perf_counter() - envelope_start
        with self._lock, torch.cuda.device(self.device):
            self._buffer_epoch += 1
            epoch = self._buffer_epoch
            before = self.scratch.snapshot()
            allocation_start = time.perf_counter()
            grew = False
            grew |= self.scratch.ensure("metadata", packet.requirements.metadata_items, torch.int64)
            grew |= self.scratch.ensure("numeric", packet.requirements.numeric_items, torch.float64)
            grew |= self.scratch.ensure("output", packet.requirements.output_items, torch.float64)
            if grew:
                torch.cuda.synchronize(self.device)
            allocation_s = time.perf_counter() - allocation_start
            after = self.scratch.snapshot()

            stage_start = time.perf_counter()
            meta_host = self.scratch.host["metadata"]
            numeric_host = self.scratch.host["numeric"]
            output_host = self.scratch.host["output"]
            meta_host[:packet.requirements.metadata_items].copy_(packet.metadata)
            numeric_host[:packet.requirements.numeric_items].copy_(packet.numeric)
            meta_host[H_BUFFER_EPOCH] = epoch
            meta_host[H_GLOBAL_STATUS:H_RECEIPT_SUM+1] = 0
            status_offset = int(meta_host[H_STATUS_OFFSET])
            status_end = int(meta_host[H_REQUEST_OFFSET])
            meta_host[status_offset:status_end] = 0
            host_stage_s = time.perf_counter() - stage_start

            meta_gpu = self.scratch.gpu["metadata"]
            numeric_gpu = self.scratch.gpu["numeric"]
            output_gpu = self.scratch.gpu["output"]
            stream = torch.cuda.current_stream(self.device)
            e0, e1, e2, e3, e4, e5, e6 = self._events
            e0.record(stream)
            begin = time.perf_counter()
            meta_gpu[:packet.requirements.metadata_items].copy_(
                meta_host[:packet.requirements.metadata_items], non_blocking=True)
            metadata_h2d_enqueue_s = time.perf_counter() - begin
            e1.record(stream)
            begin = time.perf_counter()
            numeric_gpu[:packet.requirements.numeric_items].copy_(
                numeric_host[:packet.requirements.numeric_items], non_blocking=True)
            numeric_h2d_enqueue_s = time.perf_counter() - begin
            e2.record(stream)
            h2d_sync_start = time.perf_counter()
            e2.synchronize()
            h2d_sync_s = time.perf_counter() - h2d_sync_start
            device_metadata_h2d_s = e0.elapsed_time(e1) / 1000
            device_numeric_h2d_s = e1.elapsed_time(e2) / 1000

            common = [meta_gpu, numeric_gpu, output_gpu, packet.requirements.metadata_items,
                      packet.requirements.numeric_items, packet.requirements.output_items,
                      packet.requirements.requests]
            kernel_host_start = time.perf_counter()
            self.module.launch("packet_validate", packet.requirements.requests, common)
            self.module.launch("packet_powers", packet.requirements.powers,
                               common + [packet.requirements.powers])
            self.module.launch("packet_terms", packet.requirements.coefficients,
                               common + [packet.requirements.coefficients])
            self.module.launch("packet_sum", packet.requirements.outputs,
                               common + [packet.requirements.outputs])
            kernel_enqueue_s = time.perf_counter() - kernel_host_start
            e3.record(stream)
            sync_start = time.perf_counter()
            e3.synchronize()
            kernel_and_sync_host_s = time.perf_counter() - kernel_host_start
            completion_sync_s = time.perf_counter() - sync_start
            device_kernel_s = e2.elapsed_time(e3) / 1000

            e4.record(stream)
            d2h_start = time.perf_counter()
            meta_host[:status_end].copy_(meta_gpu[:status_end], non_blocking=True)
            metadata_d2h_enqueue_s = time.perf_counter() - d2h_start
            e5.record(stream)
            copied_output_items = packet.requirements.output_items if diagnostics else 2*packet.requirements.outputs
            d2h_start = time.perf_counter()
            output_host[:copied_output_items].copy_(output_gpu[:copied_output_items], non_blocking=True)
            numeric_d2h_enqueue_s = time.perf_counter() - d2h_start
            e6.record(stream)
            d2h_sync_start = time.perf_counter()
            e6.synchronize()
            d2h_sync_s = time.perf_counter() - d2h_sync_start

            device_metadata_d2h_s = e4.elapsed_time(e5) / 1000
            device_numeric_d2h_s = e5.elapsed_time(e6) / 1000
            if int(meta_host[H_BUFFER_EPOCH]) != epoch:
                raise PacketStructureError("stale packet buffer epoch returned")
            receipts = [int(meta_host[index]) for index in range(H_RECEIPT_VALIDATE, H_RECEIPT_SUM+1)]
            if int(meta_host[H_GLOBAL_STATUS]) != 0:
                raise PacketStructureError(
                    f"device rejected packet header ({int(meta_host[H_GLOBAL_STATUS])}); "
                    f"receipts={receipts}; header={meta_host[:HEADER_WIDTH].tolist()}")
            if receipts != [1, 1, 1, 1]:
                raise PacketStructureError(f"missing packet kernel receipt: {receipts}")

            scatter_start = time.perf_counter()
            output_count = packet.requirements.outputs
            cell_count = packet.requirements.coefficients
            power_count = packet.requirements.powers
            final_lo = output_host[:output_count]
            final_hi = output_host[output_count:2*output_count]
            term_lo_base = 2*output_count
            term_hi_base = term_lo_base + cell_count
            power_lo_base = term_hi_base + cell_count
            power_hi_base = power_lo_base + power_count
            request_offset = int(meta_host[H_REQUEST_OFFSET])
            statuses = meta_host[status_offset:status_end].tolist()
            results = {}
            for b, request in enumerate(packet.requests):
                descriptor = meta_host[request_offset+b*REQUEST_WIDTH:
                                       request_offset+(b+1)*REQUEST_WIDTH].tolist()
                status = int(statuses[b])
                if status == 1:
                    result = RangeResult(request.request_id, "invalid",
                        message="nonfinite/reversed interval, nonpoint coefficient, or nonnormalized state domain")
                elif status == 2:
                    result = RangeResult(request.request_id, "masked")
                elif status == 3:
                    result = RangeResult(request.request_id, "overflow",
                        message="no finite enclosure from this arithmetic chain")
                elif status == 4:
                    result = RangeResult(request.request_id, "packet_error",
                        message="device rejected a request descriptor")
                elif status != 0:
                    result = RangeResult(request.request_id, "packet_error",
                        message=f"unknown packet status {status}")
                else:
                    output_start, outputs = descriptor[R_OUTPUT_START], descriptor[R_OUTPUT_COUNT]
                    cell_start, cells = descriptor[R_CELL_START], descriptor[R_CELL_COUNT]
                    power_start, powers = descriptor[R_POWER_START], descriptor[R_POWER_COUNT]
                    result_powers = ()
                    terms_lo = terms_hi = None
                    if diagnostics:
                        terms_lo = output_host[term_lo_base+cell_start:
                            term_lo_base+cell_start+cells].clone().reshape(outputs, descriptor[R_TERMS])
                        terms_hi = output_host[term_hi_base+cell_start:
                            term_hi_base+cell_start+cells].clone().reshape(outputs, descriptor[R_TERMS])
                        layout = _shape(request)[3]
                        result_powers = tuple((variable, exponent,
                            float(output_host[power_lo_base+power_start+i]),
                            float(output_host[power_hi_base+power_start+i]), False)
                            for i, (variable, exponent) in enumerate(layout.powers))
                    result = RangeResult(request.request_id, "ok",
                        final_lo[output_start:output_start+outputs].clone(),
                        final_hi[output_start:output_start+outputs].clone(),
                        terms_lo=terms_lo, terms_hi=terms_hi, powers=result_powers)
                results[request.request_id] = result
            scatter_s = time.perf_counter() - scatter_start
            if timings is not None:
                delta_device_allocations = after["device_allocation_operations"] - before["device_allocation_operations"]
                delta_pinned_allocations = after["pinned_allocation_operations"] - before["pinned_allocation_operations"]
                timings.update(buffer_epoch=epoch, kernel_receipt=receipts,
                    envelope_check_s=envelope_check_s,
                    host_stage_owned_payload_s=host_stage_s,
                    device_allocation_s=allocation_s,
                    metadata_h2d_enqueue_s=metadata_h2d_enqueue_s,
                    numeric_h2d_enqueue_s=numeric_h2d_enqueue_s,
                    h2d_sync_s=h2d_sync_s,
                    kernel_enqueue_s=kernel_enqueue_s,
                    kernel_and_sync_host_s=kernel_and_sync_host_s,
                    completion_sync_s=completion_sync_s,
                    metadata_d2h_enqueue_s=metadata_d2h_enqueue_s,
                    numeric_d2h_enqueue_s=numeric_d2h_enqueue_s,
                    d2h_sync_s=d2h_sync_s, scatter_and_private_wrap_s=scatter_s,
                    device_metadata_h2d_s=device_metadata_h2d_s,
                    device_numeric_h2d_s=device_numeric_h2d_s,
                    device_kernel_s=device_kernel_s,
                    device_metadata_d2h_s=device_metadata_d2h_s,
                    device_numeric_d2h_s=device_numeric_d2h_s,
                    actual_kernel_invocations=4, h2d_copy_operations=2,
                    d2h_copy_operations=2, metadata_h2d_bytes=packet.requirements.metadata_bytes,
                    numeric_h2d_bytes=packet.requirements.numeric_bytes,
                    metadata_d2h_bytes=status_end*8,
                    numeric_d2h_bytes=copied_output_items*8,
                    device_allocation_operations=delta_device_allocations,
                    pinned_allocation_operations=delta_pinned_allocations,
                    scratch=after, packet_requirements=packet.requirements.__dict__)
            return results

    def evaluate(self, requests, *, backend="cuda", diagnostics=False, timings=None):
        if backend != "cuda":
            raise ValueError("range work packets are a CUDA-only opt-in path")
        return evaluate_range_work_packets(requests, executor=self,
                                           diagnostics=diagnostics, timings=timings)


def evaluate_range_work_packets(requests: Sequence[RangeRequest], *, executor=None,
                                diagnostics=False, timings=None,
                                limits=DEFAULT_PACKET_LIMITS):
    """Validate, safely split and execute heterogeneous requests in four launches."""
    start = time.perf_counter()
    requests = tuple(requests)
    ids = [request.request_id for request in requests]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique request IDs required")
    results, gpu, fallbacks = {}, [], []
    validation_start = time.perf_counter()
    for request in requests:
        if request.cancelled or not request.enabled:
            results[request.request_id] = RangeResult(
                request.request_id, "cancelled" if request.cancelled else "masked")
            continue
        try:
            _validate_structure(request)
        except (TypeError, ValueError, IndexError) as error:
            results[request.request_id] = RangeResult(request.request_id, "invalid", message=str(error))
            continue
        (fallbacks if request.step_powers is not None else gpu).append(request)
    validation_s = time.perf_counter() - validation_start
    build_start = time.perf_counter()
    if gpu and executor is None:
        executor = RangePacketExecutor(limits=limits)
    packets, oversize = split_range_work_packets(gpu, limits=executor.limits) if gpu else ([], [])
    packet_build_s = time.perf_counter() - build_start
    packet_timings = []
    for packet in packets:
        packet_timing = {}
        results.update(executor.execute_packet(packet, diagnostics=diagnostics, timings=packet_timing))
        packet_timings.append(packet_timing)
    oversize_timings = []
    for request, message in oversize:
        # A request too large for the bounded heterogeneous scratch retains the
        # already-validated parent CUDA path.  It is explicit, never an OOM retry.
        local = {}
        results.update(evaluate_range_requests([request], backend="cuda",
                                               diagnostics=diagnostics, timings=local))
        oversize_timings.append(dict(request_id=request.request_id, reason=message, timing=local))
    fallback_timing = {}
    if fallbacks:
        results.update(evaluate_range_requests(fallbacks, backend="cuda",
                                               diagnostics=diagnostics, timings=fallback_timing))
    ended = time.perf_counter()
    if timings is not None:
        timings.update(validation_and_layout_s=validation_s,
            host_packet_build_s=packet_build_s,
            packet_count=len(packets), packet_request_counts=[p.requirements.requests for p in packets],
            packet_semantic_key_counts=[len({(r.exponents, r.domain_lo.numel(), r.kind,
                r.state_variables, r.time_variable,
                ("external", tuple(sorted(r.step_powers))) if r.step_powers is not None else ("domain",),
                tuple(r.coefficients_lo.shape), str(r.coefficients_lo.dtype),
                str(r.coefficients_lo.device)) for r in p.requests}) for p in packets],
            packet_timings=packet_timings, oversize_parent=oversize_timings,
            oversize_parent_requests=len(oversize), external_fallback_requests=len(fallbacks),
            external_fallback_timing=fallback_timing, total_s=ended-start,
            group_sizes=([p.requirements.requests for p in packets]
                         + [size for row in oversize_timings
                            for size in row["timing"].get("group_sizes", [])]),
            kernel_receipts=([p["kernel_receipt"] for p in packet_timings]
                             + [receipt for row in oversize_timings
                                for receipt in row["timing"].get("kernel_receipts", [])]),
            actual_kernel_invocations=(sum(p["actual_kernel_invocations"] for p in packet_timings)
                + sum(row["timing"].get("actual_kernel_invocations", 0)
                      for row in oversize_timings)))
        for key in ("envelope_check_s", "device_allocation_s", "host_stage_owned_payload_s",
                    "metadata_h2d_enqueue_s", "numeric_h2d_enqueue_s",
                    "h2d_sync_s",
                    "kernel_enqueue_s", "kernel_and_sync_host_s", "completion_sync_s",
                    "metadata_d2h_enqueue_s", "numeric_d2h_enqueue_s", "d2h_sync_s",
                    "scatter_and_private_wrap_s", "device_metadata_h2d_s",
                    "device_numeric_h2d_s", "device_kernel_s",
                    "device_metadata_d2h_s", "device_numeric_d2h_s"):
            timings[key] = sum(packet.get(key, 0) for packet in packet_timings)
        for key in ("h2d_copy_operations", "d2h_copy_operations",
                    "metadata_h2d_bytes", "numeric_h2d_bytes",
                    "metadata_d2h_bytes", "numeric_d2h_bytes",
                    "device_allocation_operations", "pinned_allocation_operations"):
            timings[key] = (sum(packet.get(key, 0) for packet in packet_timings)
                + sum(row["timing"].get(key, 0) for row in oversize_timings))
        # Compatibility aggregates are host spans, never labelled pure device time.
        timings["grouping_and_packing_s"] = validation_s + packet_build_s + sum(
            packet.get("envelope_check_s", 0) + packet.get("host_stage_owned_payload_s", 0)
            for packet in packet_timings)
        timings["compute_and_transfers_s"] = sum(
            packet.get("metadata_h2d_enqueue_s", 0) + packet.get("numeric_h2d_enqueue_s", 0)
            + packet.get("h2d_sync_s", 0) + packet.get("kernel_and_sync_host_s", 0)
            + packet.get("metadata_d2h_enqueue_s", 0)
            + packet.get("numeric_d2h_enqueue_s", 0) + packet.get("d2h_sync_s", 0)
            for packet in packet_timings)
        timings["scatter_and_wrap_s"] = sum(
            packet.get("scatter_and_private_wrap_s", 0) for packet in packet_timings)
        timings["fallback_s"] = fallback_timing.get("total_s", 0) + sum(
            row["timing"].get("total_s", 0) for row in oversize_timings)
        timings["h2d_and_structure_s"] = sum(
            packet.get("metadata_h2d_enqueue_s", 0) + packet.get("numeric_h2d_enqueue_s", 0)
            + packet.get("h2d_sync_s", 0) for packet in packet_timings)
        timings["kernel_and_sync_s"] = sum(
            packet.get("kernel_and_sync_host_s", 0) for packet in packet_timings)
        timings["d2h_and_checks_s"] = sum(
            packet.get("metadata_d2h_enqueue_s", 0) + packet.get("numeric_d2h_enqueue_s", 0)
            + packet.get("d2h_sync_s", 0) for packet in packet_timings)
    return results


def packet_primitive_probe(a, b, *, device="cuda:0"):
    module = get_packet_module(device)
    a = torch.tensor(a, dtype=torch.float64, device=device)
    b = torch.tensor(b, dtype=torch.float64, device=device)
    output = torch.empty((a.numel(), 4), dtype=torch.float64, device=device)
    module.launch("packet_arithmetic_probe", a.numel(), [a, b, output, a.numel()])
    return output.cpu()


_SELF_TESTED = set()
_SELF_TEST_LOCK = threading.Lock()


def packet_cuda_startup_check(executor):
    """Gate the packet module on real directed operations and a packet pow3."""
    module = executor.module
    key = (module.device, module.build["ptx_sha256"])
    with _SELF_TEST_LOCK:
        if key in _SELF_TESTED:
            return dict(reused=True, build=module.build, limits=executor.limits.__dict__)
        a = [5e-324, -5e-324, 1e-160, -1e-160, 1., 1e30]
        b = [.5, .5, 1e-160, 1e-160, 5e-324, -1e30]
        output = packet_primitive_probe(a, b, device=executor.device)
        for row, x, y in zip(output, a, b):
            for index, exact in ((0, Fraction(x)+Fraction(y)), (2, Fraction(x)*Fraction(y))):
                if not Fraction(float(row[index])) <= exact <= Fraction(float(row[index+1])):
                    raise FloatingPointError("CUDA packet primitive startup check failed")
        x = float.fromhex("0x1.7d3ecfa658d9bp+9")
        coefficient = torch.ones((1, 1), dtype=torch.float64)
        domain = torch.tensor([x], dtype=torch.float64)
        request = RangeRequest("packet-startup-pow3", ((3,),), coefficient,
                               coefficient.clone(), domain, domain.clone())
        result = executor.evaluate([request], diagnostics=True)[request.request_id]
        if (not result.ok or not Fraction(float(result.lo[0])) <= Fraction(x)**3
                <= Fraction(float(result.hi[0]))):
            raise FloatingPointError("CUDA packet power startup check failed")
        _SELF_TESTED.add(key)
        return dict(reused=False, build=module.build, limits=executor.limits.__dict__,
                    primitive_bounds=[[float(value).hex() for value in row] for row in output],
                    pow3_bounds=[float(result.lo[0]).hex(), float(result.hi[0]).hex()],
                    scratch_after_self_test=executor.scratch.snapshot())
