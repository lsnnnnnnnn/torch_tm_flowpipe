"""Versioned, fail-closed experiment protocol contracts."""

from .eligibility import (
    EligibilityDecision,
    evaluate_primary_eligibility,
    partition_and_recompute_pareto,
)
from .pareto import recompute_pareto
from .runtime import ConfigurationStepTiming, measure_configuration_step
from .schema import (
    BoundSemantics,
    FailureCategory,
    RUNTIME_BOUNDARY_VERSION,
    SCHEMA_VERSION,
)

__all__ = [
    "BoundSemantics",
    "ConfigurationStepTiming",
    "EligibilityDecision",
    "FailureCategory",
    "RUNTIME_BOUNDARY_VERSION",
    "SCHEMA_VERSION",
    "evaluate_primary_eligibility",
    "measure_configuration_step",
    "partition_and_recompute_pareto",
    "recompute_pareto",
]
