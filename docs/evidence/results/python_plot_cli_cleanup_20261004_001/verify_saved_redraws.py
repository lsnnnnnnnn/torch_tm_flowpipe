"""Redraw saved data into a new directory; no solver, MATLAB, or content digest."""
from contextlib import ExitStack
import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = REPO / "docs/evidence/results"
ARCH = EVIDENCE / "archcomp26_20261001"


def forbidden(*_args, **_kwargs):
    raise AssertionError("content digest or MATLAB export attempted")


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    dp = ARCH / "native_dp_less_full20_001/plots/dp_less_native_225x100_t_theta1_tube.geometry.json"
    tora = ARCH / "native_tora_remain_full20_001/plots/tora_native_x1_x2_endpoint.geometry.json"
    octagon = ARCH / "tm_octagon_stream_harmonic_smoke_20261002_001/harmonic_stream.geometry.json"
    stream = octagon.with_name("harmonic_stream.jsonl")
    quad = ARCH / "quad_paper_fourway_saved_20261002/quad_paper_fourway_t_x3_pooled_tube.geometry.json"
    quad_csv = quad.with_suffix(".csv")
    sources = [dp, tora, octagon, stream, quad, quad_csv]
    before = {path: path.read_bytes() for path in sources}
    checks = []
    sys.path.insert(0, str(REPO / "src"))

    with ExitStack() as guard:
        for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
            guard.enter_context(patch.object(hashlib, name, forbidden))
        plot = importlib.import_module("torch_tm_flowpipe.flowpipe_plot")
        nohash = importlib.import_module("torch_tm_flowpipe.flowpipe_plot_nohash")
        directional = importlib.import_module("torch_tm_flowpipe.tm_octagon_nohash")
        guard.enter_context(patch.object(plot, "write_matlab", forbidden))
        guard.enter_context(patch.object(directional, "_write_matlab", forbidden))

        def run(module, stem, args):
            prefix = output / stem
            argv = [*args, "--output", str(prefix)]
            started = time.perf_counter()
            with patch.object(sys, "argv", [module.__name__, *argv]):
                assert (module.main() if module is directional else module.main(argv)) == 0
            elapsed = time.perf_counter() - started
            assert not prefix.with_suffix(".m").exists()
            for suffix, signature in ((".png", b"\x89PNG\r\n\x1a\n"), (".pdf", b"%PDF-")):
                assert prefix.with_suffix(suffix).read_bytes().startswith(signature)
            receipt = json.loads(prefix.with_suffix(".render.json").read_text())
            assert '"matlab"' not in json.dumps(receipt)
            assert '"sha256"' not in json.dumps(receipt)
            checks.append({"entry": module.__name__, "argv": argv, "wall_s": elapsed,
                           "png_bytes": prefix.with_suffix(".png").stat().st_size,
                           "pdf_bytes": prefix.with_suffix(".pdf").stat().st_size,
                           "matlab_output": False})
            return prefix

        run(plot, "tora_saved_state_endpoint", ["--geometry", str(tora)])
        run(nohash, "dp_saved_time_tube", ["--geometry", str(dp)])
        run(directional, "octagon_saved_endpoint", ["--geometry", str(octagon), "--view", "endpoint"])
        streamed = run(directional, "octagon_saved_stream_tube", ["--stream", str(stream)])
        original_octagon = json.loads(before[octagon])
        rebuilt_octagon = json.loads(streamed.with_suffix(".geometry.json").read_text())
        assert rebuilt_octagon["frames"] == original_octagon["frames"]
        assert rebuilt_octagon["initial_box"] == original_octagon["initial_box"]

        original_dp = json.loads(before[dp])
        ranges = Path(original_dp["series"][0]["source_path"])
        before[ranges] = ranges.read_bytes()
        native = run(nohash, "dp_saved_native_reexport", [
            "--ranges", str(ranges), "--benchmark", "Double Pendulum",
            "--instance-id", "double-pendulum-less-robust",
            "--label", original_dp["series"][0]["label"],
            "--coordinate-names", "theta1,theta2,theta1_dot,theta2_dot",
            "--projection", "t,theta1", "--view", "tube",
            "--step-size", "0.01", "--expected-steps", "100", "--expected-lanes", "225",
            "--spec", str(REPO / "benchmarks/plot_specs/double_pendulum_less_robust_2026_nohash.json"),
        ])
        rebuilt_dp = json.loads(native.with_suffix(".geometry.json").read_text())
        assert rebuilt_dp["series"][0]["frames"] == original_dp["series"][0]["frames"]
        assert rebuilt_dp["spec"] == original_dp["spec"]
        assert rebuilt_dp["projection"] == original_dp["projection"]

        # Call only the saved-data helpers; the historical script's main also writes .m.
        spec = importlib.util.spec_from_file_location("quad_saved", quad.parent / "plot_saved_x3.py")
        saved = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(saved)
        saved.PREFIX = output / "quad_saved_fourway_t_x3"
        saved.write_matlab = forbidden
        data = json.loads(before[quad])
        saved.PREFIX.with_suffix(".geometry.json").write_bytes(before[quad])
        saved.write_csv(data)
        saved.render(data)
        assert saved.PREFIX.with_suffix(".geometry.csv").read_bytes() == before[quad_csv]
        assert saved.PREFIX.with_suffix(".geometry.json").read_bytes() == before[quad]
        with quad_csv.open(newline="") as handle:
            quad_rows = list(csv.DictReader(handle))
        assert len(quad_rows) == 2002
        assert [len(item.get("tube", [])) for item in data["series"]] == [1000, 0, 0, 1000]
        assert "torch" not in sys.modules

    assert not list(output.glob("*.m"))
    unchanged = []
    for path, raw in before.items():
        assert path.read_bytes() == raw
        unchanged.append({"path": str(path), "bytes": len(raw), "byte_equal_after": True})
    receipt = {
        "scope": "saved-data Python renderer acceptance only; no new numerical experiment or NNCS certificate",
        "python": sys.executable,
        "hash_policy": "all hashlib constructors, new and file_digest blocked during rendering",
        "solver_imported": False,
        "cli_checks": checks,
        "data_checks": {"dp_native_frames_equal": 100, "dp_lanes_per_frame": 225,
                        "stream_directional_frames_equal": 3, "quad_csv_rows_byte_equal": len(quad_rows)},
        "sources_unchanged": unchanged,
        "geometry_csv_caveat": "three CLIs use JSON; existing QUAD CSV regenerated by its saved-data helper",
        "legacy_source_export": "not executed; pre-existing identity checks remain",
    }
    (output / "ACCEPTANCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cli_checks": len(checks), "source_files_byte_unchanged": len(unchanged),
                      "matlab_files": 0, "quad_csv_rows": len(quad_rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
