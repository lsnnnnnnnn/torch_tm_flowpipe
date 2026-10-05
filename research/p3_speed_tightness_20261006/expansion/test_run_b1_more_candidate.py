"""CPU-only contract/field checks for the new Docking/SP saved comparator."""

import json
from pathlib import Path
import tempfile

from run_b1_more_candidate import compare_saved


def check():
    with tempfile.TemporaryDirectory(prefix="archcomp26-more-wrapper-") as directory:
        root = Path(directory)
        for instance in ("docking", "single-pendulum"):
            sp = instance == "single-pendulum"
            steps, events, order = (100, 50, 2) if sp else (400, 400, 3)
            reference, data = root / (instance + "-old"), root / (instance + "-new")
            for folder, wall in ((reference, 1.0), (data, 2.0)):
                folder.mkdir()
                (folder / "ranges.jsonl").write_text('{}\n' * steps)
                (folder / "safety.jsonl").write_text('{}\n' * events)
                (folder / "config.yaml").write_text('fixed: contract\n')
                (folder / "START.json").write_text(json.dumps({"driver_argv": [
                    "/frozen/driver.py", str(folder / "config.yaml"), "--metrics-json", str(folder / "metrics.json")]}))
                result = dict(status="completed", safety_events=events,
                              end_to_end_floating_point_nn_certificate=False, wall_s=wall)
                if sp:
                    result.update(completed_substeps=100, all_substeps_accepted=True,
                                  author_safe_bounds_all_nonpositive=True, independent_window_tube_boxes_safe=True)
                else:
                    result.update(observed_substeps=400, accepted_substeps=400,
                                  checker_verdict="UNKNOWN_REPORTED_BY_AUTHOR_CHECKER")
                (folder / "RESULT.json").write_text(json.dumps(result))
                (folder / "metrics.json").write_text(json.dumps(dict(
                    B=1, order=order, broken=0, steps=20 if sp else 40,
                    substeps=5 if sp else 10, elapsed_s=wall)))
            assert compare_saved(instance, reference, data)["order"] == order
            metrics = json.loads((data / "metrics.json").read_text())
            metrics["order"] += 1
            (data / "metrics.json").write_text(json.dumps(metrics))
            try:
                compare_saved(instance, reference, data)
            except RuntimeError as error:
                assert "order" in str(error)
            else:
                raise AssertionError("changed work order must fail")
    print("PASS: SP order-two/window and Docking order-three/UNKNOWN comparison guards")


if __name__ == "__main__":
    check()
