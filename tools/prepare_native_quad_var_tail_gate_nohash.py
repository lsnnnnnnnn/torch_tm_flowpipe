#!/usr/bin/env python3
"""Stage a fresh Flow* source copy with the existing two-site VAR-tail repair.

This prepares source only. It never compiles, launches a solver, or edits the
frozen input tree. Run it on the server with explicit source and output paths.
"""

import difflib
import json
from pathlib import Path
import shutil
import sys
import tempfile


EVAL_SIGNATURE = "void AST_Node<DATA_TYPE>::evaluate(TaylorModel<Interval> & result,"
REPLAY_SIGNATURE = "void AST_Node<DATA_TYPE>::evaluate_remainder(Interval & result,"
OLD_EVAL = """\tcase NODE_VAR:
\t\tif(node_value.var.type == VAR_ID)
\t\t{
\t\t\tresult = tms_of_vars[node_value.var.id];
\t\t\tresult.ctrunc_normal(step_exp_table, order);
\t\t}
"""
NEW_EVAL = """\tcase NODE_VAR:
\t\tif(node_value.var.type == VAR_ID)
\t\t{
\t\t\tresult = tms_of_vars[node_value.var.id];
\t\t\t// The polynomial tail is fixed while the input remainder is refined.
\t\t\tInterval input_remainder = result.remainder;
\t\t\tresult.remainder = Interval(0);
\t\t\tresult.ctrunc_normal(step_exp_table, order);
\t\t\tintermediate_ranges.push_back(result.remainder);
\t\t\tresult.remainder += input_remainder;
\t\t}
"""
OLD_REPLAY = """\tcase NODE_VAR:
\t\tif(node_value.var.type == VAR_ID)
\t\t{
\t\t\tresult = tms_of_vars[node_value.var.id].remainder;
\t\t}
"""
NEW_REPLAY = """\tcase NODE_VAR:
\t\tif(node_value.var.type == VAR_ID)
\t\t{
\t\t\tresult = tms_of_vars[node_value.var.id].remainder;
\t\t\tresult += *iter;
\t\t\t++iter;
\t\t}
"""


def replace_in_function(text, signature, old, new):
    if text.count(signature) != 1:
        raise ValueError(f"expected one active function: {signature}")
    start = text.index(signature)
    end = text.find("\n}\n", start)
    if end < 0:
        raise ValueError(f"missing function end: {signature}")
    end += 3
    body = text[start:end]
    if body.count(old) != 1 or new in body:
        raise ValueError(f"frozen VAR_ID text differs or is already repaired: {signature}")
    return text[:start] + body.replace(old, new, 1) + text[end:]


def patch_expression(text):
    text = replace_in_function(text, EVAL_SIGNATURE, OLD_EVAL, NEW_EVAL)
    return replace_in_function(text, REPLAY_SIGNATURE, OLD_REPLAY, NEW_REPLAY)


def check_refinement_source(text):
    real = "int Flowpipe::advance(Flowpipe & result, const std::vector<Expression<Real> > & ode,"
    interval = "int Flowpipe::advance(Flowpipe & result, const std::vector<Expression<Interval> > & ode,"
    sr = "Symbolic_Remainder & symbolic_remainder) const"
    if text.count(real) != 2:
        raise ValueError("expected ordinary and symbolic-remainder Real advance paths")
    first = text.index(real)
    second = text.index(real, first + len(real))
    ordinary = text[first:text.index(interval, first)]
    symbolic = text[second:text.index(interval, second)]
    if sr not in symbolic or sr in ordinary:
        raise ValueError("cannot identify the frozen symbolic-remainder overload")
    for name, section in (("ordinary", ordinary), ("symbolic-remainder", symbolic)):
        if section.count("bool bfinished = false;") != 1 or section.count(
            "x.Picard_ctrunc_normal_remainder(newRemainders, ode,"
        ) != 1:
            raise ValueError(f"{name} refinement loop differs from frozen source")


def self_test():
    original = (
        EVAL_SIGNATURE + "\n{\n" + OLD_EVAL + "}\n"
        + REPLAY_SIGNATURE + "\n{\n" + OLD_REPLAY + "}\n"
    )
    patched = patch_expression(original)
    assert NEW_EVAL in patched and NEW_REPLAY in patched
    try:
        patch_expression(patched)
    except ValueError:
        pass
    else:
        raise AssertionError("already-repaired input must be rejected")
    print("self-test passed: exact two-site replacement and repeat rejection")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_native_quad_var_tail_gate_nohash.py FROZEN_FLOWSTAR_DIR NEW_GATE_DIR")
    source = Path(sys.argv[1]).resolve(strict=True)
    output = Path(sys.argv[2]).resolve()
    if not source.is_dir() or output.exists() or source == output or source in output.parents:
        raise SystemExit("source must be a directory and output must be a new, separate directory")
    if not output.parent.is_dir():
        raise SystemExit("output parent must already exist")
    required = ("expression.h", "Continuous.cpp", "TaylorModel.h", "Makefile", "libflowstar.a")
    if any(not (source / name).is_file() for name in required):
        raise SystemExit("frozen Flow* source, Makefile, or archive is missing")

    original = (source / "expression.h").read_text()
    patched = patch_expression(original)
    check_refinement_source((source / "Continuous.cpp").read_text())
    taylor = (source / "TaylorModel.h").read_text()
    if "expansion.ctrunc_normal(I, step_exp_table, order);\n\tremainder += I;" not in taylor:
        raise SystemExit("TaylorModel polynomial-tail accounting differs from frozen source")

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as scratch:
        staged = Path(scratch) / output.name
        copied = staged / "flowstar-toolbox"
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns("*.o", "libflowstar.a", ".git"))
        (copied / "expression.h").write_text(patched)
        diff = difflib.unified_diff(
            original.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile="frozen/expression.h", tofile="isolated/expression.h",
        )
        (staged / "expression.patch").write_text("".join(diff))
        (staged / "PREPARE.json").write_text(json.dumps({
            "source_directory": str(source),
            "new_directory": str(output),
            "changed_file": "flowstar-toolbox/expression.h",
            "change": "cache VAR polynomial truncation tail and replay it with each refined input remainder",
            "continuous_refinement_loops": "retained unchanged in copied Continuous.cpp",
            "compiled": False,
            "experiment_run": False,
            "content_digest_performed": False,
        }, indent=2) + "\n")
        staged.rename(output)
    print(output)


if __name__ == "__main__":
    main()
