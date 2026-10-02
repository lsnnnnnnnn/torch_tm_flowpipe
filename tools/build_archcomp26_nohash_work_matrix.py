#!/usr/bin/env python3
"""Build the path-attributed 2026 ARCH-COMP work matrix without content digests."""

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("benchmarks/archcomp26/manifest.json")
ATTEMPTS = Path("docs/evidence/archcomp26_nohash_attempts_20261001.json")
STEM = Path("docs/evidence/archcomp26_nohash_work_matrix_20261001")
INSTANCE_ALIASES = {
    "double-pendulum-less-robust-continuous": "double-pendulum-less-robust",
    "double-pendulum-more-robust-continuous": "double-pendulum-more-robust",
}
METHOD_ALIASES = {"pytorch_gpu_p3_port": "pytorch_gpu"}
STATUS = {
    "completed": "completed",
    "completed_numeric_unknown_property": "completed",
    "running": "running",
    "incomplete_unknown": "early_stopped",
    "early_stopped_numerical_rejection": "early_stopped",
    "early_stopped_checker_unsafe": "early_stopped",
    "failed_before_plant_advance": "failed",
    "failed_before_ode": "failed",
    "incomplete_rejected_first_substep": "failed",
    "failed_observer_invalid_interval": "failed",
    "incomplete_failed_contraction": "failed",
    "incomplete_unknown_and_failed_contraction": "failed",
    "failed_diagnostic_hook_after_prefix": "failed",
    "completed_short_prefix": "short_prefix_only",
}
LABELS = {
    "completed": "完成",
    "running": "运行中",
    "early_stopped": "早停",
    "failed": "失败",
    "not_attempted": "未尝试",
    "short_prefix_only": "仅短前缀",
}


def main():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    source = json.loads((ROOT / ATTEMPTS).read_text(encoding="utf-8"))
    ids = [item["id"] for item in manifest["instances"]]
    methods = manifest["methods"]
    assert len(ids) == manifest["scope"]["expected_instance_count"] == 16
    assert len(ids) == len(set(ids)) and len(methods) == len(set(methods)) == 4
    assert source["schema"] == "archcomp26-nohash-attempt-index-v1"

    grouped = defaultdict(list)
    for number, attempt in enumerate(source["attempts"], 1):
        instance = INSTANCE_ALIASES.get(attempt["instance"], attempt["instance"])
        method = METHOD_ALIASES.get(attempt["method"], attempt["method"])
        if instance not in ids or method not in methods:
            raise ValueError(f"unknown manifest cell in attempt {number}: {instance}/{method}")
        if attempt["status"] not in STATUS:
            raise ValueError(f"unknown attempt status in attempt {number}: {attempt['status']}")
        grouped[instance, method].append((number, attempt))

    cells = []
    for instance in ids:
        for method in methods:
            records = grouped[instance, method]
            # Keep explicitly supplemental numerical profiles out of the frozen main cell.
            main_records = [(n, a) for n, a in records if a.get("matrix_eligible", True)]
            substantive = [(n, a) for n, a in main_records if a["status"] != "completed_short_prefix"]
            chosen = (substantive or main_records)[-1] if main_records else None
            attempt = chosen[1] if chosen else None
            work_status = STATUS[attempt["status"]] if attempt else "not_attempted"
            if work_status == "completed" and attempt.get("complete_numerical_run") is not True:
                raise ValueError(f"completion without full-run flag: {instance}/{method}")
            if work_status == "short_prefix_only":
                limitation = "Only a short numerical prefix is recorded; no full-horizon result. Inspect the attempt for initial-set coverage."
            elif work_status == "completed":
                limitation = "At least one complete numerical run; repeated attempts, if any, are in the attempt index. No end-to-end NNCS proof."
            elif work_status == "running":
                limitation = "Dated attempt index records an in-progress run; no final time, width, or verdict in this index."
            elif work_status == "early_stopped":
                limitation = "Stopped before the requested horizon; no full-horizon time or width."
            elif work_status == "failed":
                limitation = "Diagnostic attempt did not finish the requested horizon; no comparable result."
            else:
                limitation = "No new attempt in this index; historical frozen runs are excluded."
            cells.append({
                "instance_id": instance,
                "method": method,
                "work_status": work_status,
                "attempt_count": len(records),
                "selected_attempt_number": chosen[0] if chosen else None,
                "selected_attempt_status": attempt["status"] if attempt else None,
                "selected_run_dir": attempt["run_dir"] if attempt else None,
                "selected_wall_s": attempt.get("wall_s") if attempt else None,
                "selected_property_verdict": attempt.get("author_property_verdict") if attempt else None,
                "complete_numerical_run": work_status == "completed",
                "four_way_ranking_eligible": False,
                "qualification_note": limitation,
                "attempts": [
                    {
                        "index": n,
                        "status": a["status"],
                        "role": a["role"],
                        "run_dir": a["run_dir"],
                        "local_summary": a.get("local_summary"),
                    }
                    for n, a in records
                ],
            })

    assert len(cells) == 64
    counts = Counter(c["work_status"] for c in cells)
    assert sum(counts.values()) == 64
    assert sum(c["attempt_count"] for c in cells) == len(source["attempts"])
    payload = {
        "schema": "archcomp26-nohash-work-matrix-v1",
        "date": source["date"],
        "source_paths": [str(MANIFEST), str(ATTEMPTS)],
        "source_binding": "Path attribution only; no content digest computed.",
        "scope": "New 2026 no-hash attempts only; independent of the historical frozen matrix.",
        "instance_count": len(ids),
        "method_count": len(methods),
        "cell_count": len(cells),
        "attempt_count": len(source["attempts"]),
        "status_counts": {key: counts[key] for key in LABELS},
        "methods": methods,
        "cells": cells,
    }

    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    csv_buffer = io.StringIO(newline="")
    columns = [
        "instance_id", "method", "work_status", "attempt_count",
        "selected_attempt_number", "selected_attempt_status", "selected_run_dir",
        "selected_wall_s", "selected_property_verdict",
        "complete_numerical_run", "four_way_ranking_eligible", "qualification_note",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: cell[key] for key in columns} for cell in cells)
    csv_text = csv_buffer.getvalue()

    lines = [
        "# ARCH-COMP26 无哈希工作状态矩阵（2026-10-01）",
        "",
        f"来源：`{MANIFEST}` 与 `{ATTEMPTS}`。仅按路径归属，不做内容摘要；与历史冻结的 `benchmarks/archcomp26/execution_matrix.json` 独立。",
        "",
        f"16 个实例 × 4 种方法 = {len(cells)} 个单元；本轮索引 {len(source['attempts'])} 条尝试。"
        + "；".join(f"{LABELS[key]} {counts[key]}" for key in ("completed", "running", "early_stopped", "failed", "short_prefix_only", "not_attempted")) + "。",
        "",
        "| 实例 | P3 GPU | Huan | Xiangru | FlowStar native |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_key = {(c["instance_id"], c["method"]): c for c in cells}
    for instance in ids:
        lines.append("| " + " | ".join([instance] + [LABELS[by_key[instance, method]["work_status"]] for method in methods]) + " |")
    lines += [
        "",
        "“完成”仅表示所选尝试记录了完整数值时域，包括 Docking 原生数值完成但性质 UNKNOWN/外层 exit 2 的独立状态；“早停”包含 DP more 原生的 UNKNOWN、Huan/Xiangru 的 checker Unsafe、P3 第 9 小步的数值拒绝，以及 Balancing raw4 Huan/P3 各自的数值拒绝；“失败”包括 P3 DP less 诊断和 Airplane 完整初盒首步入口/资源失败。“仅短前缀”仍不覆盖完整时域，例如 TORA reach-tanh Huan 官方文件 profile 的本轮新诊断仅有完整初盒一期、50 小步；同合同旧四方全程结果另按历史证据审计，未重复运行。较早 smoke 和各尝试原始状态保存在 JSON。",
        "Balancing Huan/Xiangru/P3/原生的早停属于明确命名的固定仓库四原态 `balancing-fixed-repo-raw4` profile。Huan/Xiangru/P3 分别在第 99/99/87 小步拒绝；原生另立修正入口在第 21 周期前三段后、第 84 小步返回 `UNCOMPLETED_SAFE`，保存 83 段。Xiangru 已接受步的保存 endpoint 在 189 处超出同小步 tube 至多 2.49e-14；论文五特征控制器仍缺，不可把这些状态当作论文主合同结果。Airplane P3 回调 trace 把原数值 profile 的首步拒绝定位为自映射收缩失败；仅放宽 x/y/z 余项初猜的 profile 首步引擎接受后，观察器因 endpoint 越出 tube 1–2 个 binary64 相邻值而拒绝，0 个可用保存步，也无全时安全结论。NAV standard 当前 P3 与原生各自有独立 640 盒×600 小步全程运行和扫描，NAV robust 当前 P3 另有独立 25 盒×600 小步全程运行和扫描；旧 NAV `ours` 是另一代引擎，其同合同记录只作历史审计，不并入新 attempt。",
        "TORA reach-tanh 当前 working P3 已另立 0.5 秒首周期门检和 500 步完整数值作业；旧 `engine_linear_leaf_v2` P3 同合同全程仅保留作历史对照，不填本轮新 attempt。作者 Huan 的新一期仍仅为短前缀，其旧同合同全程另列历史证据。",
        "TORA remain 新的 Huan 第 190 步只读追踪和仅改 x2 余项初猜的第 192 步早停均作为补充诊断登记，`matrix_eligible=false`；本格仍选择原 `h=0.1` 统一余项全时尝试。Airplane 二分 64 盒规划目前只运行两个角盒各一个小步；高角盒数值接受后性质 Unknown，不能替代完整初盒失败格。",
        "",
        "所有 64 个单元在本索引中均不具备四方法排名资格。重复次数按各条 attempt 计，不能从完成状态推断计时资格；保存的数值结果不构成端到端浮点 NNCS 证明。标为运行中的尝试尚无终点时间或宽度；DP more 早停没有完整 T=0.4 结果。",
        "",
        f"重算：`python3 tools/{Path(__file__).name}`。脚本只读取上述两个 JSON，写此 JSON/CSV/Markdown，并检查 JSON/CSV 各有 64 个数据单元。",
        "",
    ]
    (ROOT / STEM.with_suffix(".json")).write_text(json_text, encoding="utf-8")
    (ROOT / STEM.with_suffix(".csv")).write_text(csv_text, encoding="utf-8")
    (ROOT / STEM.with_suffix(".md")).write_text("\n".join(lines), encoding="utf-8")

    assert len(json.loads((ROOT / STEM.with_suffix(".json")).read_text(encoding="utf-8"))["cells"]) == 64
    with (ROOT / STEM.with_suffix(".csv")).open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 64
    print("Wrote independent no-hash matrix: 64 JSON cells, 64 CSV data rows")


if __name__ == "__main__":
    main()
