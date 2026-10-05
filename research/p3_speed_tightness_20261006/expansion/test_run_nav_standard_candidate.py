"""CPU-only synthetic saved-output comparison check; no solver or old checker."""

import json
from pathlib import Path
import tempfile

from run_nav_standard_candidate import compare_saved


def check():
    with tempfile.TemporaryDirectory(prefix="archcomp26-nav-wrapper-") as directory:
        reference, data = Path(directory) / "old", Path(directory) / "new"
        for folder, wall in ((reference, 1.0), (data, 2.0)):
            folder.mkdir()
            with (folder / "ranges.bin").open("wb") as stream:
                stream.truncate(640 * 600 * 136)
            rows = [dict(substep=i, accepted_boxes=640, rejected_lanes=[],
                         saved_range_records=640, solver_status_counts={"0": 640},
                         obstacle_intersecting_saved_tubes=0) for i in range(1, 601)]
            (folder / "observations.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (folder / "initial_boxes.json").write_text('fixture: same ledger')
            (folder / "config.yaml").write_text('fixture: same config')
            (folder / "START.json").write_text(json.dumps({"driver_argv": [
                "/frozen/driver.py", str(folder / "config.yaml"), "--metrics-json", str(folder / "metrics.json")]}))
            (folder / "RESULT.json").write_text(json.dumps(dict(
                status="completed", observed_substeps=600, accepted_lane_substeps=384000,
                range_records=384000, full_horizon_covered=True, full_initial_set_covered=True,
                author_verdict_lines=["VERIFIED"], end_to_end_floating_point_nn_certificate=False,
                run_id=folder.name, wall_s=wall)))
            (folder / "metrics.json").write_text(json.dumps(dict(
                B=640, order=3, broken=0, steps=30, substeps=20, elapsed_s=wall)))
        assert compare_saved(reference, data)["range_bytes_equal"] == 52224000
        with (data / "ranges.bin").open("r+b") as stream:
            stream.write(b'1')
        try:
            compare_saved(reference, data)
        except RuntimeError as error:
            assert "record 1" in str(error)
        else:
            raise AssertionError("changed range byte must fail")
    print("PASS: new NAV 640 x 600 comparison, runtime metadata exclusion and changed-byte rejection")


if __name__ == "__main__":
    check()
