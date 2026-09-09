"""Targeted semantic tampering, after refreshing the byte manifest."""
from copy import deepcopy
import pytest

from experiments.range_batch_device.common import RUN,read,save,read_records,digest
from experiments.range_batch_device.verify import check_record,write_manifest,manifest
from experiments.range_batch_device.derive import check_time_row,check_invocations,decision


@pytest.fixture(scope="module")
def real_entries():
    inputs={r["request"]["request_id"]:r for r in read_records(RUN/"raw/corpus/independent.jsonl.gz")}
    for output in read_records(RUN/"raw/operator_outputs/cpu.jsonl.gz"):
        record=inputs[output["request_id"]]
        if len(record["request"]["exponents"])>1 and output["result"]["powers"]:
            candidate=deepcopy(record)
            candidate["request"]["exponents"].reverse()
            candidate["request_numeric_sha256"]=digest(candidate["request"])
            out=deepcopy(output);out["input_sha256"]=digest(candidate["request"])
            try:check_record(candidate,out)
            except AssertionError:return record,output
    raise AssertionError("need an actual ordered-support counterfactual")


@pytest.mark.parametrize("field",["request_id","support_order","coefficient","mask","power_error"])
def test_request_semantic_tamper_after_rehash(tmp_path,real_entries,field):
    record,output=deepcopy(real_entries)
    save(tmp_path/"input.json",record);save(tmp_path/"output.json",output)
    check=lambda:check_record(read(tmp_path/"input.json"),read(tmp_path/"output.json"))
    check()
    if field=="request_id":
        record["request"]["request_id"]="wrong-task"
        output["request_id"]=output["result"]["request_id"]="wrong-task"
    elif field=="support_order":record["request"]["exponents"].reverse()
    elif field=="coefficient":
        record["request"]["coefficients_lo"]["values"][0]=(1e100).hex()
        record["request"]["coefficients_hi"]["values"][0]=(1e100).hex()
    elif field=="mask":record["request"]["enabled"]=False
    else:output["result"]["powers"][0][2:4]=["0x0.0p+0","0x0.0p+0"]
    record["request_numeric_sha256"]=digest(record["request"]);output["input_sha256"]=digest(record["request"])
    save(tmp_path/"input.json",record);save(tmp_path/"output.json",output)
    write_manifest(tmp_path);manifest(tmp_path)
    with pytest.raises(AssertionError):check()


def test_device_invocation_tamper_after_rehash(tmp_path):
    value=read(RUN/"cuda_actual_invocations.json");check_invocations(value)
    value["audit_device_executed_kernel_invocations"]=0
    save(tmp_path/"invocations.json",value);write_manifest(tmp_path);manifest(tmp_path)
    with pytest.raises(AssertionError):check_invocations(read(tmp_path/"invocations.json"))


def test_timing_classification_tamper_after_rehash(tmp_path):
    row=next(r for r in read(RUN/"raw/timing/samples.json") if r["mode"]=="GPU_resident")
    check_time_row(row)
    row["mode"]="GPU_full_roundtrip"
    save(tmp_path/"event.json",row);write_manifest(tmp_path);manifest(tmp_path)
    with pytest.raises(AssertionError):check_time_row(read(tmp_path/"event.json"))


def test_conclusion_tamper_after_rehash(tmp_path):
    expected=decision(RUN)
    assert read(RUN/"RESULT.json")==expected
    changed=deepcopy(expected)
    changed["status"]="RANGE_BATCH_CORRECT__NO_MATERIAL_DEVICE_GAIN" if expected["status"].endswith("USEFUL") else "RANGE_BATCH_CPU_CLOSED__CUDA_LOCAL_PILOT_USEFUL"
    save(tmp_path/"RESULT.json",changed);write_manifest(tmp_path);manifest(tmp_path)
    with pytest.raises(AssertionError):assert read(tmp_path/"RESULT.json")==decision(RUN)
