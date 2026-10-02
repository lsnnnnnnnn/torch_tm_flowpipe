"""Make one isolated diagnostic Flow* source variant; never edit the input."""
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: patch_native_flowstar_refinement_probe_nohash.py SOURCE NEW_OUTPUT")
source, output = map(Path, sys.argv[1:])
if source.resolve() == output.resolve() or output.exists():
    raise SystemExit("source and output must differ; output must not exist")
code = source.read_text()
start = code.index("int Flowpipe::advance(Flowpipe & result, const std::vector<Expression<Real> > & ode,")
end = code.index("int Flowpipe::advance(Flowpipe & result, const std::vector<Expression<Interval> > & ode,", start)
section = code[start:end]
needle = "\tbool bfinished = false;"
if section.count(needle) != 1:
    raise SystemExit("expected exactly one non-SR Real advance refinement loop")
section = section.replace(needle,
    "\tbool bfinished = true; // isolated probe: keep initial Picard truncation remainder")
output.write_text(code[:start] + section + code[end:])
