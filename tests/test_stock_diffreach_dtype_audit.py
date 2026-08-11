from __future__ import annotations

from pathlib import Path

from experiments.run_stock_diffreach_vdp_reproduction import _builder_dtype_audit


def test_stock_builder_dtype_audit_records_implicit_and_float32_sites(tmp_path: Path) -> None:
    files = (
        "run_dyn.py",
        "src/reachability.py",
        "src/symbolic_remainder.py",
        "src/interval.py",
        "src/polynomial.py",
        "src/taylor_model.py",
        "src/rhs_eval.py",
    )
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "run_dyn.py":
            path.write_text("value = jnp.array([1.0])\n", encoding="utf-8")
        else:
            path.write_text(
                "def builder(dtype=jnp.float32):\n    return dtype\n",
                encoding="utf-8",
            )
    audit = _builder_dtype_audit(tmp_path)
    assert audit["classification"] == "mixed_builder_dtype"
    assert any(
        row["declared_dtype"] == "implicit_default_under_jax_x64"
        for row in audit["sites"]
    )
    assert any(
        row["declared_dtype"] == "jnp.float32_default_or_literal"
        for row in audit["sites"]
    )
    assert len(audit["files"]) == len(files)
