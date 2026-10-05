#!/usr/bin/env python3
"""Synthetic saved-record parser/width check; no solver or old checker."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile


def prohibited(*a, **kw):
    raise RuntimeError("content digest prohibited")


for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
    if hasattr(hashlib, name):
        setattr(hashlib, name, prohibited)

import yaml
from run_tanh_cutoff_candidate import compare_widths, RECORD


def main():
    with tempfile.TemporaryDirectory(prefix="new-tanh-cutoff-check-") as directory:
        root = Path(directory)
        old, new = root / "old_fixture", root / "new_fixture"
        for path, cutoff in ((old, 1e-6), (new, 1e-8)):
            path.mkdir()
            cfg = dict(cut_off_threshold=cutoff, model_dir=str(path/"controller_plant_u.onnx"),
                       output_scale=1, output_offset=0)
            (path/"config.yaml").write_text(yaml.safe_dump(cfg))
            (path/"controller_plant_u.onnx").write_bytes(b"synthetic-byte-identity-only")
            (path/"controller_plant_u.onnx.json").write_text(json.dumps(dict(
                model_path=str(path/"controller_plant_u.onnx"), mat_path=str(path/"model.mat"),
                activations=["relu"]*3+["tanh"], scale=11, offset=0)))
            (path/"START.json").write_text(json.dumps(dict(driver_argv=["driver",str(path/"config.yaml")])))
            (path/"observations.jsonl").write_text("".join(json.dumps(dict(
                substep=i,accepted=True,interval_valid=True))+"\n" for i in range(1,501)))
            (path/"metrics.json").write_text(json.dumps(dict(B=1,order=3,steps=10,substeps=50,broken=0)))
            (path/"RESULT.json").write_text(json.dumps(dict(status="completed_full_numerical_horizon",
                accepted_substeps=500,property_evaluated=False,wall_s=1)))
            endpoint = [[-.05,.15],[-.85,-.65],[0,0],[-.5,.5]] if path == old else [[-.02,.12],[-.8,-.7],[0,0],[-.5,.5]]
            tube = [-1,1] if path == old else [-.8,.8]
            with (path/"ranges.bin").open("wb") as stream:
                for step in range(1,501):
                    stream.write(RECORD.pack(0,step,.01,*[v for pair in endpoint for v in (*tube,*pair)]))
        receipt = compare_widths(old,new,("controller_plant_u.onnx",))
        assert receipt["absolute_width_rows"] == 4000 and receipt["strictly_narrower_observations"] == 3000
        assert receipt["subset_observations"] == 4000 and receipt["all_widths_nonincreasing"]
        assert receipt["property_record"]["terminal_box_in_target"] is True
        assert receipt["property_record"]["target_contained_saved_endpoint_steps"] == list(range(1,501))
        zero = next(r for r in receipt["full_state_full_step_widths"] if (r["state"],r["geometry"]) == ("x3","endpoint"))
        assert zero["equal"] == 500 and zero["final_width_ratio"] is None
        assert len(list(csv.DictReader((root/"ABSOLUTE_WIDTHS.csv").open()))) == 4000
        model = json.loads((new/"controller_plant_u.onnx.json").read_text())
        model["activations"][-1] = "sigmoid"
        (new/"controller_plant_u.onnx.json").write_text(json.dumps(model))
        try:
            compare_widths(old,new,("controller_plant_u.onnx",))
        except RuntimeError as error:
            assert "tanh" in str(error)
        else:
            raise AssertionError("wrong activation receipt accepted")
    result = dict(status="PASS", scope="synthetic four-state 500-step binary records and metadata only",
                  width_rows=4000, narrower=3000, equal=1000, subset=4000,
                  checks=["tube/endpoint indexing", "exact width classification", "zero-width ratio",
                          "all target endpoint observations", "tanh-versus-sigmoid receipt refusal"],
                  real_nn_or_solver_run=False, old_checker_executed=False, digest_operations=0)
    Path(__file__).with_name("TANH_CUTOFF_CPU_CHECK.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
