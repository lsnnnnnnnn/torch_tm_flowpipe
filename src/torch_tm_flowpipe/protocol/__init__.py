"""Versioned, fail-closed experiment protocol contracts."""

from .backend_identity import (
    AUDIT_BEHAVIOR_ENV_VARS,
    BackendIdentityError,
    FlowstarBackendIdentity,
    classify_flowstar_backend,
    enabled_audit_behavior_variables,
    inspect_diagnostic_flowstar_backend,
    inspect_primary_flowstar_backend,
)

from .eligibility import (
    EligibilityDecision,
    evaluate_primary_eligibility,
    partition_and_recompute_pareto,
)
from .config import configuration_semantics, expected_configuration_rows
from .carry import projected_affine_box_reset
from .pareto import recompute_pareto
from .runtime import ConfigurationStepTiming, measure_configuration_step
from .schema import (
    BoundKind,
    BoundSemantics,
    FailureCategory,
    RefinementSemantics,
    RUNTIME_BOUNDARY_VERSION,
    SCHEMA_VERSION,
)

__all__ = [
    "AUDIT_BEHAVIOR_ENV_VARS",
    "BackendIdentityError",
    "BoundKind",
    "BoundSemantics",
    "ConfigurationStepTiming",
    "EligibilityDecision",
    "FailureCategory",
    "FlowstarBackendIdentity",
    "RUNTIME_BOUNDARY_VERSION",
    "RefinementSemantics",
    "SCHEMA_VERSION",
    "evaluate_primary_eligibility",
    "classify_flowstar_backend",
    "configuration_semantics",
    "expected_configuration_rows",
    "enabled_audit_behavior_variables",
    "inspect_diagnostic_flowstar_backend",
    "inspect_primary_flowstar_backend",
    "projected_affine_box_reset",
    "measure_configuration_step",
    "partition_and_recompute_pareto",
    "recompute_pareto",
]
