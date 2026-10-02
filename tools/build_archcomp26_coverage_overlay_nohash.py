#!/usr/bin/env python3
"""Overlay audited historical full runs on the separate new-attempt matrix."""

import csv
import io
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parents[2] / "results/archcomp_review_20260923"
MATRIX = ROOT / "docs/evidence/archcomp26_nohash_work_matrix_20261001.json"
TANH_AUDIT = ROOT / "docs/evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/HISTORICAL_REUSE_AUDIT.json"
OUT = ROOT / "docs/evidence/archcomp26_coverage_overlay_20261002"
NAV_AUDITS = {
    "huan": "docs/ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md",
    "xiangru": "docs/ARCHCOMP26_NAV_XIANGRU_HISTORICAL_CONTRACT_AUDIT_20261002.md",
    "flowstar_native": "docs/ARCHCOMP26_NAV_P3_NATIVE_HISTORICAL_AUDIT_20261002.md",
}
NAV_RESULTS = {
    ("nav-standard", "huan"): HIST / "evidence_v1/suite_v1/nav_standard_huan/result.json",
    ("nav-standard", "xiangru"): HIST / "evidence_v1/suite_v1/nav_standard_xiangru/result.json",
    ("nav-robust", "huan"): HIST / "evidence_v2/timing_v1/nav_robust_r2_huan/result.json",
    ("nav-robust", "xiangru"): HIST / "evidence_v1/suite_v1/nav_robust_xiangru/result.json",
    ("nav-robust", "flowstar_native"): ROOT / "docs/evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/result.json",
}
LABELS = {
    "current_numeric_full": "新全程",
    "historical_same_contract_numeric_full": "旧全程",
    "contract_blocked": "合同阻塞",
    "no_full_numeric_evidence": "无全程",
}
METHOD_LABELS = {
    "pytorch_gpu": "P3 GPU",
    "huan": "Huan",
    "xiangru": "Xiangru",
    "flowstar_native": "FlowStar native",
}


def relative(path):
    return os.path.relpath(path, OUT.parent)


def historical_sources():
    sources = {}
    for (instance, method), path in NAV_RESULTS.items():
        sources[instance, method] = {
            "result_path": str(path),
            "audit_path": NAV_AUDITS[method],
            "contract_scope": "fixed official 2026 NAV controller and author executable state order; the separate author-repository Git LFS model identity is unestablished",
            "expected_steps": 600,
            "expected_accepted": 384000 if instance == "nav-standard" else 15000,
        }
    tanh = json.loads(TANH_AUDIT.read_text(encoding="utf-8"))
    for method, name in (("pytorch_gpu", "p3"), ("huan", "huan"), ("xiangru", "xiangru"), ("flowstar_native", "native")):
        sources["tora-reach-tanh", method] = {
            "result_path": tanh["historical_full_numerical_runs"][name]["result_path"],
            "audit_path": str(TANH_AUDIT.relative_to(ROOT)),
            "contract_scope": "fixed official 2026 MAT ReLU/ReLU/ReLU/tanh controller with plant u=11f; the paper's sigmoid-hidden prose is a different, unresolved identity",
            "expected_steps": 500,
            "expected_accepted": 500,
        }
    assert len(sources) == 9
    for (instance, method), source in sources.items():
        result = json.loads(Path(source["result_path"]).read_text(encoding="utf-8"))
        assert result["status"] == "completed" and result["complete"] is True, (instance, method)
        if method == "flowstar_native":
            periods = 30 if instance.startswith("nav-") else 10
            assert result["control_periods_started"] == periods, (instance, method)
            assert result["range_records"] == source["expected_accepted"], (instance, method)
        else:
            assert result["attempted_steps"] == source["expected_steps"], (instance, method)
            assert result["accepted_lane_steps"] == source["expected_accepted"], (instance, method)
        source["recorded_status"] = result["status"]
        source["recorded_complete"] = result["complete"]
    return sources


def main():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["cell_count"] == 64
    sources = historical_sources()
    cells = []
    for cell in matrix["cells"]:
        instance, method = cell["instance_id"], cell["method"]
        source = sources.get((instance, method))
        if cell["work_status"] == "completed":
            coverage = "current_numeric_full"
        elif source:
            assert cell["work_status"] in {"short_prefix_only", "not_attempted"}
            coverage = "historical_same_contract_numeric_full"
        elif instance == "airplane-discrete" and cell["work_status"] == "not_attempted":
            coverage = "contract_blocked"
        else:
            coverage = "no_full_numeric_evidence"
        cells.append({
            "instance_id": instance,
            "method": method,
            "new_work_status": cell["work_status"],
            "new_selected_run_dir": cell["selected_run_dir"],
            "coverage_status": coverage,
            "historical_result_path": source["result_path"] if source else None,
            "historical_audit_path": source["audit_path"] if source else None,
            "historical_contract_scope": source["contract_scope"] if source else None,
            "historical_recorded_status": source["recorded_status"] if source else None,
            "historical_recorded_complete": source["recorded_complete"] if source else None,
        })
    counts = Counter(cell["coverage_status"] for cell in cells)
    assert len(cells) == 64 and sum(counts.values()) == 64
    payload = {
        "schema": "archcomp26-coverage-overlay-v1",
        "date": "2026-10-02",
        "source_matrix": str(MATRIX.relative_to(ROOT)),
        "source_historical_audits": sorted({source["audit_path"] for source in sources.values()}),
        "new_attempt_count": matrix["attempt_count"],
        "cell_count": len(cells),
        "coverage_counts": dict(counts),
        "qualification": "Full means recorded numerical horizon under each cell's named contract; it does not establish that every participant model identity matches the 2026 paper, an end-to-end floating-point NNCS certificate, or timing-rank qualification. Historical runs remain outside the new attempt index.",
        "cells": cells,
    }
    OUT.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = list(cells[0])
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(cells)
    OUT.with_suffix(".csv").write_text(buf.getvalue(), encoding="utf-8")

    by_key = {(cell["instance_id"], cell["method"]): cell for cell in cells}
    instances = list(dict.fromkeys(cell["instance_id"] for cell in cells))
    methods = matrix["methods"]
    lines = [
        "# ARCH-COMP26 16×4 新尝试与历史全程覆盖附表",
        "",
        f"从[本轮工作矩阵]({relative(MATRIX)})的 **{matrix['attempt_count']} 条新尝试**生成；旧运行只在下方有原始结果与合同审计可对应时覆盖缺口，不计入新尝试。全程表示数值时域完成，不表示性质证明或速度排名资格。",
        "",
        f"**64 格：新全程 {counts['current_numeric_full']}；同合同旧全程 {counts['historical_same_contract_numeric_full']}；无全程 {counts['no_full_numeric_evidence']}；Airplane discrete 合同阻塞 {counts['contract_blocked']}。**",
        "",
        "| 实例 | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |",
        "| --- | " + " | ".join("---" for _ in methods) + " |",
    ]
    for instance in instances:
        lines.append("| " + instance + " | " + " | ".join(LABELS[by_key[instance, method]["coverage_status"]] for method in methods) + " |")
    lines += [
        "",
        "“无全程”保留本轮原始早停或失败，具体状态和路径见[工作矩阵]({})；“合同阻塞”仅指 Airplane discrete 四格，目前缺 2026 参与者权威离散转移及控制更新顺序，见[执行门](../ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md)。新全程也只在各自具名合同内成立：Single Pendulum 当前是两物理态加辅助时钟，官方第三态执行身份仍未建立。".format(relative(MATRIX)),
        "",
        f"## {counts['historical_same_contract_numeric_full']} 条同合同历史全程来源",
        "",
        "| 实例 | 方法 | 本轮状态 | 历史原始结果 | 合同审计 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        if cell["coverage_status"] != "historical_same_contract_numeric_full":
            continue
        result = Path(cell["historical_result_path"])
        audit = ROOT / cell["historical_audit_path"]
        lines.append("| {} | {} | {} | [result]({}) | [审计]({}) |".format(
            cell["instance_id"], METHOD_LABELS[cell["method"]], cell["new_work_status"],
            relative(result), relative(audit)))
    lines += [
        "",
        "NAV 的历史合同是固定官方 point/set ONNX 加作者可执行 `[x,y,v,θ]` 顺序；作者另一仓库 Git LFS 模型与官方模型的二进制身份尚未建立。Huan/Xiangru NAV 的相应保存范围使用共享数值核心且直接逐字节相同，不能当作两份独立证明。旧 NAV `ours` 的全程记录另见[审计](../ARCHCOMP26_NAV_P3_NATIVE_HISTORICAL_AUDIT_20261002.md)，本附表的 P3 格优先采用当前工作 P3 新全程。",
        "",
        "TORA reach-tanh 的旧四方只复用于固定官方 ReLU³/tanh、`u=11f` 合同；论文合并文字的 sigmoid 隐层是来源冲突。原生旧 `VERIFIED` 是作者终点 checker 标签，不能提升为独立端到端浮点 NNCS 证明。Airplane discrete 的两个方法外 CPU 前缀诊断也不填四方格。",
        "",
        "重算：`python3 tools/build_archcomp26_coverage_overlay_nohash.py`。脚本只读本轮矩阵、九条旧 result 和已保存的 TORA 复用审计，不启动实验；生成本页、[CSV]({})及[JSON]({})。".format(OUT.with_suffix(".csv").name, OUT.with_suffix(".json").name),
        "",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    assert len(json.loads(OUT.with_suffix(".json").read_text(encoding="utf-8"))["cells"]) == 64
    with OUT.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 64
    print("Coverage overlay:", dict(counts))


if __name__ == "__main__":
    main()
