from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import PROTOCOL_B, PROTOCOL_C, load_spec
from run_flowstar import render_cpp


def test_diffreach_adapter_calls_upstream_class_method_without_copied_core() -> None:
    path = HERE / "run_diffreach.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "DiffReachPlantCore" not in class_names
    assert "StrictAffineDiffReachPlantCore" not in class_names
    assert "UPSTREAM_STEP_ONCE = CT_DYN_REACH.step_once" in source
    assert "UPSTREAM_STEP_ONCE(core, carry, None)" in source
    assert "from src.picard import remainder_picard" in source
    assert "from src.taylor_model import QuadTM" in source


def test_flowstar_common_box_and_native_carry_are_distinct() -> None:
    spec = load_spec()
    system = spec["systems"]["riccati"]
    kwargs = {
        "h": 0.01,
        "horizon": 0.02,
        "remainder_estimation": 1e-4,
        "cutoff": 1e-15,
    }
    box_source = render_cpp(system, protocol=PROTOCOL_B, **kwargs)
    native_source = render_cpp(system, protocol=PROTOCOL_C, **kwargs)
    assert "current = Flowpipe(endpoint_box);" in box_source
    assert "current = next;" not in box_source
    assert "current = next;" in native_source
    assert "current = Flowpipe(endpoint_box);" not in native_source
    for source in (box_source, native_source):
        assert "const unsigned int local_order = 2;" in source
        assert "setFixedStepsize" in source
        assert "order1_supported" in source
        assert "next.tmvPre.tms[state].remainder" in source


def test_torch_common_box_has_no_inflation_or_affine_generator_carry() -> None:
    source = (HERE / "run_torch.py").read_text(encoding="utf-8")
    assert "current_box = list(segment.final_tm.range_box())" in source
    assert "inflate(" not in source
    assert 'protocol == PROTOCOL_B' in source
