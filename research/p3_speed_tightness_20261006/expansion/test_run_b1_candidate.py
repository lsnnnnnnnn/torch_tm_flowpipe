"""CPU-only checks of the new wrapper's staging/comparison; no numerical imports."""

import json
from pathlib import Path
import tempfile

from run_b1_candidate import SOURCE_FILES, compare_saved, stage


def check():
    with tempfile.TemporaryDirectory(prefix="archcomp26-expansion-wrapper-") as directory:
        root = Path(directory)
        for instance in SOURCE_FILES:
            source, staged = root / (instance + "-source"), root / (instance + "-staged")
            source.mkdir()
            for name in SOURCE_FILES[instance]:
                (source / name).write_bytes(("fixture:" + name).encode())
            if instance == "tora-tanh":
                (source / "config.yaml").write_text("model_dir: /saved/tanh/controller_plant_u.onnx\n")
                (source / "controller_plant_u.onnx.json").write_text(json.dumps({
                    "model_path": "/saved/tanh/controller_plant_u.onnx",
                    "mat_path": "/saved/tanh/nn_tora_relu_tanh.mat"}))
            receipt = stage(instance, source, staged)
            assert set(receipt["byte_equal_copies"]) == set(SOURCE_FILES[instance])
            for name in SOURCE_FILES[instance]:
                assert (source / name).read_bytes() == (staged / name).read_bytes()
            if instance == "tora-tanh":
                assert str(staged) in (staged / "config.yaml").read_text()
                assert "/saved/tanh" in (source / "config.yaml").read_text()
            try:
                stage(instance, source, staged)
            except FileExistsError:
                pass
            else:
                raise AssertionError("existing output must never be replaced")

        reference, data = root / "reference", root / "candidate"
        for folder, timing in ((reference, 1.0), (data, 2.0)):
            folder.mkdir()
            (folder / "ranges.bin").write_bytes(b"\0" * (60 * 216))
            (folder / "observations.jsonl").write_text('{"accepted": true}\n' * 60)
            (folder / "config.yaml").write_text("fixture: unchanged\n")
            (folder / "START.json").write_text(json.dumps({"driver_argv": [
                "/frozen/driver.py", str(folder / "config.yaml"), "--metrics-json", str(folder / "metrics.json")]}))
            (folder / "RESULT.json").write_text(json.dumps({
                "status": "completed", "accepted_substeps": 60, "all_substeps_accepted": True,
                "saved_tubes_box_disjoint_official_unsafe": True, "end_to_end_strict_certificate": False,
                "wall_s": timing, "driver_elapsed_s": timing}))
            (folder / "metrics.json").write_text(json.dumps({
                "B": 1, "order": 3, "broken": 0, "steps": 30, "substeps": 2,
                "elapsed_s": timing, "signed_zero": -0.0}))
        result = compare_saved("attitude", reference, data)
        assert result["completed_substeps"] == 60 and result["old_checker_executed"] is False
        metrics = json.loads((data / "metrics.json").read_text())
        metrics["signed_zero"] = 0.0
        (data / "metrics.json").write_text(json.dumps(metrics))
        try:
            compare_saved("attitude", reference, data)
        except RuntimeError as error:
            assert "signed_zero" in str(error)
        else:
            raise AssertionError("scientific signed-zero change must be detected")
    print("PASS: new wrapper staging, no overwrite, artifact relocation and saved-field comparison")


if __name__ == "__main__":
    check()
