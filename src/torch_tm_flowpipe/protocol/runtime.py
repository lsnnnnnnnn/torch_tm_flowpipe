from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


EngineResult = TypeVar("EngineResult")
CompletionResult = TypeVar("CompletionResult")


@dataclass(frozen=True)
class ConfigurationStepTiming(Generic[EngineResult, CompletionResult]):
    engine_result: EngineResult
    completion_result: CompletionResult
    total_seconds: float
    engine_seconds: float


def measure_configuration_step(
    engine_call: Callable[[], EngineResult],
    completion_call: Callable[[EngineResult], CompletionResult],
    *,
    synchronize: Callable[[], None] | None = None,
) -> ConfigurationStepTiming[EngineResult, CompletionResult]:
    """Measure the engine and the full carried-state step boundary.

    The total boundary includes endpoint materialization, range evaluation,
    projection, reset, and carry construction performed by ``completion_call``.
    An optional synchronizer makes asynchronous device work part of the same
    boundary.
    """
    sync = synchronize or (lambda: None)
    sync()
    total_started = time.perf_counter()
    engine_started = time.perf_counter()
    engine_result = engine_call()
    sync()
    engine_seconds = time.perf_counter() - engine_started
    completion_result = completion_call(engine_result)
    sync()
    total_seconds = time.perf_counter() - total_started
    return ConfigurationStepTiming(
        engine_result=engine_result,
        completion_result=completion_result,
        total_seconds=total_seconds,
        engine_seconds=engine_seconds,
    )
