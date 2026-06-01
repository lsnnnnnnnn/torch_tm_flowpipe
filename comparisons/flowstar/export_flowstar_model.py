"""Export plant-only polynomial ODE benchmarks to Flow* inputs.

The default target is the current ``chenxin415/flowstar`` toolbox repository,
which exposes Flow* as a C++ static library rather than as a standalone model
file executable.  A legacy ``.model`` renderer is kept for older Flow* 2.x style
installations, but the comparison script defaults to toolbox C++.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Mapping

try:  # PyYAML is a runtime dependency for the comparison suite.
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only on incomplete envs.
    raise SystemExit("PyYAML is required: python -m pip install PyYAML") from exc


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Flow* config must be a YAML mapping: {path}")
    required = ["system", "state_vars", "initial", "ode", "h", "steps", "order"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"missing required keys in {path}: {missing}")
    return data


def _num(x: Any) -> str:
    """Format a Python/YAML scalar for Flow* without accidental YAML artifacts."""
    val = float(x)
    if math.isfinite(val):
        return f"{val:.17g}"
    raise ValueError(f"non-finite numeric value: {x!r}")


def _cpp_num(x: Any) -> str:
    text = _num(x)
    # Keep integer-looking values valid as C++ doubles where needed.
    return text


def _flowstar_expr(expr: str) -> str:
    """Translate a tiny Python-style polynomial expression into Flow* parser syntax."""
    # Flow* expression strings use ^ for powers.  The configs use ^ already for
    # readability; this also accepts accidental Python **.
    return str(expr).replace("**", "^")


def _cpp_string(text: str) -> str:
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _safe_cpp_identifier(text: str) -> str:
    ident = re.sub(r"\W+", "_", text.strip())
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    return ident


def _safe_file_stem(text: str | Path) -> str:
    # Do not use Path.stem blindly: benchmark names contain decimal points
    # such as h0.01, and Path.stem would truncate them to h0.
    raw = str(text)
    name = Path(raw).name
    for suffix in (".cpp", ".model", ".plt", ".txt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return stem or "flowstar_case"


def _flowstar_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("flowstar", {}))


def _time_var(config: Mapping[str, Any]) -> str:
    cfg = _flowstar_cfg(config)
    preferred = str(cfg.get("time_var", "t"))
    state_vars = set(map(str, config["state_vars"]))
    if preferred not in state_vars:
        return preferred
    i = 0
    while f"{preferred}_{i}" in state_vars:
        i += 1
    return f"{preferred}_{i}"


def render_toolbox_cpp(
    config: Mapping[str, Any],
    *,
    h: float,
    steps: int,
    order: int,
    output_prefix: str = "flowstar_case",
    fixed_step: bool = True,
    fixed_order: bool = True,
    remainder_radius: float | None = None,
    cutoff: float | None = None,
) -> str:
    """Render a C++ benchmark for the current Flow* toolbox repository.

    The generated program follows the style of ``chenxin415/flowstar`` examples:
    include ``Continuous.h``, declare variables, construct an ``ODE`` from string
    derivatives, use ``Computational_Setting::setFixedStepsize``, call
    ``ode.reach(...)``, and emit GNUPLOT interval files for parsing.
    """
    if not fixed_step or not fixed_order:
        raise ValueError("first comparison suite only supports fixed step and fixed order")

    state_vars = [str(v) for v in config["state_vars"]]
    initial = dict(config["initial"])
    ode = dict(config["ode"])
    flowstar_cfg = _flowstar_cfg(config)
    metric_vars = [str(v) for v in config.get("metric_vars", state_vars)]
    plot_vars = [str(v) for v in flowstar_cfg.get("plot_vars", metric_vars)]
    if not plot_vars:
        plot_vars = state_vars[:1]

    missing_init = [v for v in state_vars if v not in initial]
    missing_ode = [v for v in state_vars if v not in ode]
    if missing_init or missing_ode:
        raise ValueError(f"missing init vars {missing_init} or ode vars {missing_ode}")

    time_var = _time_var(config)
    all_vars = state_vars + [time_var]
    time_horizon = float(h) * int(steps)
    rem = float(remainder_radius if remainder_radius is not None else flowstar_cfg.get("remainder_estimation", 1.0e-10))
    cutoff_value = float(cutoff if cutoff is not None else flowstar_cfg.get("cutoff", 1.0e-15))
    symbolic_remainder_size = int(flowstar_cfg.get("symbolic_remainder_size", 0))

    exprs = [_flowstar_expr(str(ode[v])) for v in state_vars] + ["1"]
    expr_list = ", ".join(_cpp_string(e) for e in exprs)

    lines: list[str] = []
    lines.append('#include "Continuous.h"')
    lines.append("#include <ctime>")
    lines.append("#include <cstdio>")
    lines.append("#include <vector>")
    lines.append("using namespace flowstar;")
    lines.append("using namespace std;")
    lines.append("")
    lines.append("int main()")
    lines.append("{")
    lines.append("  Variables vars;")
    for var in all_vars:
        cid = _safe_cpp_identifier(var)
        lines.append(f"  int {cid}_id = vars.declareVar({_cpp_string(var)});")
    lines.append("")
    lines.append(f"  ODE<Real> ode({{{expr_list}}}, vars);")
    lines.append("  Computational_Setting setting(vars);")
    lines.append(f"  setting.setFixedStepsize({_cpp_num(h)}, {int(order)});")
    lines.append(f"  setting.setCutoffThreshold({_cpp_num(cutoff_value)});")
    lines.append("  vector<Interval> remainder_estimation(vars.size());")
    lines.append("  for(unsigned int i = 0; i < vars.size(); ++i)")
    lines.append("  {")
    lines.append(f"    remainder_estimation[i] = Interval({_cpp_num(-abs(rem))}, {_cpp_num(abs(rem))});")
    lines.append("  }")
    lines.append("  setting.setRemainderEstimation(remainder_estimation);")
    lines.append("  setting.printOn();")
    lines.append("")
    lines.append("  vector<Interval> box(vars.size());")
    for var in state_vars:
        cid = _safe_cpp_identifier(var)
        lo, hi = initial[var]
        lines.append(f"  box[{cid}_id] = Interval({_cpp_num(lo)}, {_cpp_num(hi)});")
    tcid = _safe_cpp_identifier(time_var)
    lines.append(f"  box[{tcid}_id] = Interval(0.0, 0.0);")
    lines.append("  Flowpipe initialSet(box);")
    lines.append("  vector<Constraint> safeSet;")
    lines.append("  Result_of_Reachability result;")
    lines.append("")
    lines.append("  clock_t begin, end;")
    lines.append("  begin = clock();")
    if symbolic_remainder_size > 0:
        lines.append(f"  Symbolic_Remainder sr(initialSet, {symbolic_remainder_size});")
        lines.append(f"  ode.reach(result, initialSet, {_cpp_num(time_horizon)}, setting, safeSet, sr);")
    else:
        lines.append(f"  ode.reach(result, initialSet, {_cpp_num(time_horizon)}, setting, safeSet);")
    lines.append("  end = clock();")
    lines.append("  printf(\"FLOWSTAR_RUNTIME_S %.17g\\n\", (double)(end - begin) / CLOCKS_PER_SEC);")
    lines.append("  printf(\"FLOWSTAR_COMPLETED %d\\n\", result.isCompleted() ? 1 : 0);")
    lines.append("  printf(\"FLOWSTAR_SAFE %d\\n\", result.isSafe() ? 1 : 0);")
    lines.append("  printf(\"FLOWSTAR_UNSAFE %d\\n\", result.isUnsafe() ? 1 : 0);")
    lines.append("  if(!result.isCompleted())")
    lines.append("  {")
    lines.append('    printf("Flowpipe computation is terminated due to the large overestimation.\\n");')
    lines.append("  }")
    lines.append("")
    lines.append("  result.transformToTaylorModels(setting);")
    lines.append("  Plot_Setting plot_setting(vars);")
    lines.append("  plot_setting.printOn();")
    for var in plot_vars:
        if var not in state_vars and var != time_var:
            continue
        safe_name = f"{output_prefix}_{time_var}_{var}"
        lines.append(f"  plot_setting.setOutputDims({_cpp_string(time_var)}, {_cpp_string(var)});")
        lines.append(f"  plot_setting.plot_2D_interval_GNUPLOT(\"./\", {_cpp_string(safe_name)}, result.tmv_flowpipes, setting);")
        lines.append(f"  printf(\"FLOWSTAR_PLOT {safe_name} {time_var} {var}\\n\");")
    lines.append("  return 0;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def render_legacy_model(
    config: Mapping[str, Any],
    *,
    h: float,
    steps: int,
    order: int,
    output_name: str = "flowpipes.plt",
    fixed_step: bool = True,
    fixed_order: bool = True,
) -> str:
    """Render an older Flow* 2.x-style ``.model`` file.

    This is retained as a fallback for environments that still expose a Flow*
    model-file parser executable.  It is *not* the default for the current
    ``chenxin415/flowstar`` toolbox repository.
    """
    if not fixed_step or not fixed_order:
        raise ValueError("first comparison suite only supports fixed step and fixed order")
    state_vars = list(config["state_vars"])
    initial = dict(config["initial"])
    ode = dict(config["ode"])
    flowstar_cfg = _flowstar_cfg(config)
    plot_vars = list(flowstar_cfg.get("plot_vars", state_vars[: min(2, len(state_vars))]))
    if not plot_vars:
        plot_vars = state_vars[:1]
    if len(plot_vars) == 1:
        gnuplot_line = f"gnuplot interval {plot_vars[0]}"
    else:
        gnuplot_line = f"gnuplot octagon {plot_vars[0]},{plot_vars[1]}"

    missing_init = [v for v in state_vars if v not in initial]
    missing_ode = [v for v in state_vars if v not in ode]
    if missing_init or missing_ode:
        raise ValueError(f"missing init vars {missing_init} or ode vars {missing_ode}")

    time_horizon = float(h) * int(steps)
    rem = flowstar_cfg.get("remainder_estimation", 1.0e-10)
    cutoff = flowstar_cfg.get("cutoff", 1.0e-15)
    precision = int(flowstar_cfg.get("precision", 53))

    lines: list[str] = []
    lines.append("continuous reachability")
    lines.append("{")
    lines.append(f"  state var {','.join(state_vars)}")
    lines.append("  setting")
    lines.append("  {")
    lines.append(f"    fixed steps {_num(h)}")
    lines.append(f"    time {_num(time_horizon)}")
    lines.append(f"    remainder estimation {_num(rem)}")
    lines.append("    identity precondition")
    lines.append(f"    {gnuplot_line}")
    lines.append(f"    fixed orders {int(order)}")
    lines.append(f"    cutoff {_num(cutoff)}")
    lines.append(f"    precision {precision}")
    lines.append(f"    output flowpipes \"{output_name}\"")
    lines.append("    print on")
    lines.append("  }")
    lines.append("  poly ode 1")
    lines.append("  {")
    for var in state_vars:
        lines.append(f"    {var}' = {_flowstar_expr(ode[var])}")
    lines.append("  }")
    lines.append("  init")
    lines.append("  {")
    for var in state_vars:
        lo, hi = initial[var]
        lines.append(f"    {var} in [{_num(lo)}, {_num(hi)}]")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# Backwards-compatible name used by earlier tests/callers.  It now returns the
# toolbox C++ source, matching the current chenxin415/flowstar repository.
def render_model(
    config: Mapping[str, Any],
    *,
    h: float,
    steps: int,
    order: int,
    output_name: str = "flowpipes.plt",
    fixed_step: bool = True,
    fixed_order: bool = True,
) -> str:
    output_prefix = _safe_file_stem(output_name) if output_name else "flowstar_case"
    return render_toolbox_cpp(
        config,
        h=h,
        steps=steps,
        order=order,
        output_prefix=output_prefix,
        fixed_step=fixed_step,
        fixed_order=fixed_order,
    )


def export_model(
    config_path: str | Path,
    output_path: str | Path,
    *,
    h: float,
    steps: int,
    order: int,
    plot_output_name: str | None = None,
    target: str = "toolbox_cpp",
    remainder_radius: float | None = None,
    cutoff: float | None = None,
) -> Path:
    cfg = load_config(config_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if target == "toolbox_cpp":
        prefix = _safe_file_stem(plot_output_name or out.stem)
        text = render_toolbox_cpp(cfg, h=h, steps=steps, order=order, output_prefix=prefix, remainder_radius=remainder_radius, cutoff=cutoff)
    elif target == "legacy_model":
        plot_name = plot_output_name or (out.with_suffix(".plt").name)
        text = render_legacy_model(cfg, h=h, steps=steps, order=order, output_name=plot_name)
    else:
        raise ValueError(f"unknown Flow* export target: {target}")
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a plant-only benchmark to Flow* input syntax.")
    parser.add_argument("config", help="YAML benchmark config")
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--output", required=True, help="Flow* C++ or legacy .model output path")
    parser.add_argument("--plot-output-name", default=None)
    parser.add_argument("--target", choices=["toolbox_cpp", "legacy_model"], default="toolbox_cpp")
    parser.add_argument("--flowstar-remainder-radius", type=float, default=None)
    parser.add_argument("--flowstar-cutoff", type=float, default=None)
    args = parser.parse_args()
    path = export_model(
        args.config,
        args.output,
        h=args.h,
        steps=args.steps,
        order=args.order,
        plot_output_name=args.plot_output_name,
        target=args.target,
        remainder_radius=args.flowstar_remainder_radius,
        cutoff=args.flowstar_cutoff,
    )
    print(path)


if __name__ == "__main__":
    main()
