#!/usr/bin/env python3
"""Index existing 16x4 receipts, contracts and delivery artifacts without solving."""

import csv
import json
from collections import Counter
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/evidence"
RESULTS = EVIDENCE / "results/archcomp26_20261001"
REMOTE = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/"
IDS = ("acc-safe-distance", "airplane-continuous", "airplane-discrete", "attitude-control-avoid",
       "balancing-reach", "docking-constraint", "double-pendulum-less-robust",
       "double-pendulum-more-robust", "nav-standard", "nav-robust", "quad-reach",
       "single-pendulum-reach", "tora-remain", "tora-reach-sigmoid", "tora-reach-tanh", "unicycle-reach")
CONTRACT_NOTES = (
    "participant-order 完整初盒；T=5 全时安全距离裕量非负",
    "官方完整十二态初盒；T=2；四方失败或数值首拒，分盒首步仅为补充",
    "参与者权威离散转移及 NN 取样/更新顺序缺失；四格未运行",
    "修正非空六维 unsafe 盒；T=3 全时避开；旧空集 checker 结果不复用",
    "论文五特征控制器缺失；四方尝试仅属具名 fixed-repo-raw4 补充合同，均早停",
    "完整四态初盒；T=40 全时径向速度约束；数值完整，性质 UNKNOWN",
    "官方 less 控制器；225 初盒；T=1；四态全时安全盒",
    "官方 more 控制器；225 初盒；T=.4；四方性质/数值早停，无完整终点",
    "固定官方 point ONNX；640 初盒；T=6 避障及终点目标；新旧完整分列",
    "固定官方 set ONNX；25 初盒；T=6 避障及终点目标；新旧完整分列",
    "用户选择论文十二态方程；1024 初盒×1000 步；T=5；旧作者方程另列",
    "具名两物理态＋辅助时钟；T=1；[.5,1] 安全窗；官方第三态执行材料缺失",
    "固定 h=.1、12 初盒、T=20 全时安全；h=.05 完整补充不得替换主格",
    "用户选择官方 sigmoid 模型、u=11f；完整初盒；T=5 数值终点观察",
    "官方 ReLU³/tanh、u=11f；当前 working P3＋三方同合同历史完整",
    "用户选择论文方程；w仅入速度导数且每轨迹常值；T=10；终点仅为到达充分观察",
)
LIMITS = (
    "保存盒与作者标签不构成独立 NNCS 证明；共享主机重复计时仅描述统计。",
    "完整初盒四方无全时流管；原生64盒仅首步、单高角首期4/10接受，不能补成主格。",
    "需权威离散状态转移及控制更新顺序；连续结果和 CPU 诊断不能填四方法格。",
    "二维图不代替六维危险盒判交；所有方法缺独立端到端证书。",
    "需论文五输入控制器或权威映射；raw4 P3/Huan/Xiangru/native的首拒均在性质窗前。",
    "四方保守盒约束为 UNKNOWN；需要更强耦合分析或可信反例，不能按完整数值时域写 SAFE。",
    "Huan/Xiangru个别 endpoint 末位超出同段 tube；两种界分别保留；P3 split4缺端到端证书。",
    "P3/Huan/Xiangru至72/80，native至64/80；作者 Unsafe/Unknown不是独立真实轨迹反例。",
    "作者另一 Git LFS 模型身份未建立；旧我方引擎仅作代际附图，混合年代不作速度排名。",
    "作者另一 Git LFS 模型身份未建立；历史完整保持历史资格，不重跑来伪造新成绩。",
    "缺权威全时 reach-and-remain checker；Huan/Xiangru从未保存逐步坐标；原生生产门关闭。",
    "缺官方第三态初值、重置和 MATLAB 闭环/checker 提交入口；具名两态不冒名三态复现。",
    "Huan/Xiangru h=.1主格不完整，安全共同前缀184；不存在可启用的既有自映射重试开关。",
    "GPU未运行作者性质checker；原生仅终点VERIFIED；完整重复时间不证明稳定排名或独立证书。",
    "GPU无作者性质verdict；历史原生仅终点VERIFIED；新旧并列不得推导稳定速度排名。",
    "Huan/Xiangru终点UNKNOWN；全保存endpoint未全入目标不证明整个时间窗不可达。",
)


def read(path):
    return json.loads(path.read_text())


def relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def saved_artifacts():
    projection = "report_saved_projections_20261004_001/"
    nav = "nav_current_p3_saved_20261004_001/"
    nav_states = "nav_fourstate_saved_20261004_001/"
    tanh = "tora_tanh_current_p3_saved_20261004_001/"
    balancing = "balancing_raw4_fourway_saved_20261004_001/output/"
    dp_more = "dp_more_fourway_saved_20261004_001/"
    widths = {
        "acc-safe-distance": ["acc_fourway_saved_ranges_20261001/acc_t5_endpoint_and_full_tube_long.csv"],
        "attitude-control-avoid": [projection + "absolute_widths.csv", "attitude_corrected_fourway_campaign_20261002_001/RUNS.csv"],
        "balancing-reach": [balancing + "absolute_widths.csv", balancing + "saved_bounds.csv"],
        "docking-constraint": ["docking_fourway_saved_widths_20261002.csv", projection + "absolute_widths.csv"],
        "double-pendulum-less-robust": ["dp_less_fourway_split4_20261002/endpoint_stats.csv", "dp_less_fourway_split4_20261002/tube_union.csv"],
        "double-pendulum-more-robust": [dp_more + "absolute_widths.csv", dp_more + "saved_bounds.csv"],
        "nav-standard": [nav_states + "absolute_widths.csv", nav_states + "all_states_saved_curves.csv"],
        "nav-robust": [nav_states + "absolute_widths.csv", nav_states + "all_states_saved_curves.csv"],
        "quad-reach": ["quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv", "native_quad_paper_full50_001/SCAN.json"],
        "single-pendulum-reach": ["sp_two_state_fourway_saved_20261002/endpoint_and_window_widths.csv", "sp_two_state_fourway_saved_20261002/saved_tubes.csv"],
        "tora-remain": ["tora_remain_fourway_common_prefix_20261002/RANGES.csv"],
        "tora-reach-sigmoid": ["tora_reach_sigmoid_official2026_mat_u11_terminal_fourway.csv"],
        "tora-reach-tanh": [tanh + "absolute_widths.csv", tanh + "saved_bounds.csv"],
        "unicycle-reach": ["unicycle_paper_speed_w_constant_v1/terminal_fourway.csv"],
    }
    plots = {
        "acc-safe-distance": ["plots/nohash_saved_20261001/acc_four_method_t_safe_distance_margin_tube.png"],
        "attitude-control-avoid": [projection + "attitude_fourway_t_x4.png"],
        "balancing-reach": [balancing + "balancing_raw4_fourway_common_prefix.png"],
        "docking-constraint": [projection + "docking_fourway_t_sx.png", "plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.png"],
        "double-pendulum-less-robust": ["dp_less_fourway_split4_20261002/fourway_tube_union.png"],
        "double-pendulum-more-robust": [dp_more + "dp_more_fourway_common_prefix.png"],
        "nav-standard": [nav + "nav_saved_xy_tubes_with_working_p3.png", nav + "nav_each_method_width_difference_vs_native_with_working_p3.png"],
        "nav-robust": [nav + "nav_saved_xy_tubes_with_working_p3.png", nav + "nav_each_method_width_difference_vs_native_with_working_p3.png"],
        "quad-reach": ["quad_paper_fourway_saved_20261002/quad_paper_fourway_t_x3_pooled_tube.png", projection + "quad_paper_fourway_endpoint_x1_x3.png"],
        "single-pendulum-reach": ["sp_two_state_fourway_saved_20261002/fourway_saved_bounds.png"],
        "tora-remain": ["tora_remain_fourway_common_prefix_20261002/plots/tora_remain_2026_fourway_t_x4_saved_tube.png"],
        "tora-reach-sigmoid": ["tora_reach_sigmoid_u11_fourway_saved_figure_20261002/tora_reach_sigmoid_u11_fourway_saved.png"],
        "tora-reach-tanh": [tanh + "tora_tanh_current_fourway.png", tanh + "tora_tanh_p3_generations.png"],
        "unicycle-reach": ["unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/unicycle_paper_speed_fourway_saved.png"],
    }
    return ({k: [relative(RESULTS / p) for p in v] for k, v in widths.items()},
            {k: [relative(RESULTS / p) for p in v] for k, v in plots.items()})


def current_receipts(attempt):
    if not attempt:
        return []
    run = attempt["run_dir"]
    if not run.startswith(REMOTE):
        raise ValueError(f"Unexpected current run root: {run}")
    base = RESULTS / run[len(REMOTE):]
    if not base.exists():
        if attempt.get("local_result"):
            base = REPO / attempt["local_result"]
            base = base.parent
        else:
            base = (REPO / attempt["local_summary"]).parent / Path(run).name
    receipts = []
    for subdir in ("", "supervisor", "outer", "attempt", "run", "data", "payload", "attempt/data"):
        folder = base / subdir
        for name in ("RESULT.json", "SUPERVISOR_RESULT.json"):
            path = folder / name
            if not path.exists():
                continue
            data = read(path)
            receipts.append({
                "local_result": relative(path),
                "remote_result": (run + "/" + str(path.relative_to(base)) if name == "RESULT.json" else None),
                "local_start": relative(folder / "START.json") if (folder / "START.json").exists() else None,
                "receipt_status": data.get("status"), "receipt_exit_code": data.get("exit_code", data.get("returncode")),
                "receipt_property_fields": {key: data[key] for key in
                                            ("author_checker_lines", "author_verdict_lines", "property_verdict", "property_evaluated")
                                            if key in data},
                "scope": ("Original receipt mirror; outer and payload status must be read separately."
                          if name == "RESULT.json" else
                          "Locally named supervisor mirror; original remote filename is not established here."),
            })
    if not receipts:
        raise ValueError(f"No local selected receipt: {run}")
    return receipts


def main():
    attempts_path = EVIDENCE / "archcomp26_nohash_attempts_20261001.json"
    matrix_path = EVIDENCE / "archcomp26_nohash_work_matrix_20261001.json"
    overlay_path = EVIDENCE / "archcomp26_coverage_overlay_20261002.json"
    report_path = REPO / "docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md"
    attempts = read(attempts_path)["attempts"]
    matrix, overlay = read(matrix_path), read(overlay_path)
    if matrix["attempt_count"] != len(attempts) or overlay["new_attempt_count"] != len(attempts):
        raise ValueError("Update the source matrix/overlay before rebuilding the delivery index")
    coverage = {(c["instance_id"], c["method"]): c for c in overlay["cells"]}
    sections = {int(match.group(1)): match.group(2) for match in re.finditer(
        r"(?ms)^## (\d+)\. (.*?)(?=^## |\Z)", report_path.read_text())}
    widths, plots = saved_artifacts()
    cells = []
    for source in matrix["cells"]:
        instance, method = source["instance_id"], source["method"]
        number = IDS.index(instance) + 1
        section = sections[number]
        report_links = [str(Path("docs") / link.split("#")[0]) for _, link in re.findall(r"\[([^]]+)\]\(([^)]+)\)", section)
                        if not link.startswith(("http", "#"))]
        contracts = list(dict.fromkeys(p for p in report_links if not p.startswith("docs/evidence/")))
        old = coverage[(instance, method)]
        selected = source["selected_attempt_number"]
        attempt = attempts[selected - 1] if selected else None
        historical = old["coverage_status"] == "historical_same_contract_numeric_full"
        full = old["coverage_status"] in ("historical_same_contract_numeric_full", "current_numeric_full")
        missing = []
        if instance == "airplane-continuous":
            missing += ["主合同0段有效接受，无全时宽度或流管；64盒首步补充不能混入主图。"]
        if instance == "airplane-discrete":
            missing += ["合同阻塞：四方法没有权威离散时序，时间/宽度/图不适用。"]
        if instance == "balancing-reach":
            missing += ["raw4共同接受前缀仅83步/.415秒；已列四态图与每盒宽度，T=10未保存，论文五输入合同仍缺。"]
        if instance == "double-pendulum-more-robust":
            missing += ["已列四态共同数值前缀64步/.32秒与安全前缀60步/.30秒；T=.4终点未保存，不外推。"]
        if instance == "quad-reach":
            missing += ["Huan/Xiangru无逐步坐标范围，故其tube/全过程每盒统计不可得；终点按自己的driver对象列示。"]
        if not full and instance != "airplane-discrete":
            missing += ["未完整运行的终点和全程时间留空；只保留真实停止前缀/失败进程时间。"]
        cell = {
            "instance_id": instance, "method": method, "report_section": number,
            "contract_note": CONTRACT_NOTES[number - 1], "contract_paths": contracts,
            "coverage_status": old["coverage_status"], "current_work_status": source["work_status"],
            "current_matrix_attempt_references": source["attempts"],
            "complete_named_numerical_horizon": full,
            "selected_current_attempt_number": selected, "selected_current_run_dir": source["selected_run_dir"],
            "selected_current_receipts": current_receipts(attempt),
            "historical_full_result_path": old["historical_result_path"] if historical else None,
            "historical_same_contract_audit": old["historical_audit_path"] if historical else None,
            "property_classification_from_matrix": source["selected_property_verdict"],
            "property_classification_qualification": ("Derived current-attempt matrix classification, not a verbatim author checker label. "
                                                       "Historical numerical completion does not supply a new author verdict; consult original receipt/log and contract audit."),
            "independent_end_to_end_floating_nncs_certificate": False,
            "property_and_evidence_limits": LIMITS[number - 1],
            "time_reference": (old["historical_result_path"] if historical else relative(matrix_path)),
            "time_qualification": "No extrapolation; failed/short wall is not full-horizon time; mixed generations are not a ranking.",
            "absolute_width_references": widths.get(instance, []), "figure_references": plots.get(instance, []),
            "artifact_source_rule": "Each width/figure file retains its own source run; it may differ from the matrix-selected timing sample.",
            "missing_or_noncomparable": missing,
            "report_evidence_paths": list(dict.fromkeys(report_links)),
        }
        for p in contracts + cell["absolute_width_references"] + cell["figure_references"]:
            if not (REPO / p).exists():
                raise ValueError(f"Missing delivery reference: {p}")
        if historical and not Path(old["historical_result_path"]).exists():
            raise ValueError("Historical receipt missing")
        cells.append(cell)
    if len(cells) != 64 or len({(c["instance_id"], c["method"]) for c in cells}) != 64:
        raise ValueError("Delivery index is not a unique 16x4 grid")
    payload = {
        "schema": "archcomp26-delivery-index-v1", "as_of": "2026-10-04",
        "source_paths": [relative(p) for p in (attempts_path, matrix_path, overlay_path, report_path)],
        "source_attempt_count": len(attempts), "cell_count": 64,
        "coverage_counts": dict(Counter(c["coverage_status"] for c in cells)),
        "purpose": "Read-only evidence/artifact inventory; no new benchmark attempt and no promotion to proof.",
        "report_link_policy": "Paths are attributed directly; no content digest or identity verification is performed.",
        "cells": cells,
    }
    prefix = EVIDENCE / "archcomp26_delivery_index_20261004"
    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    keys = list(cells[0])
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for cell in cells:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in cell.items()})
    text = ["# ARCH-COMP26：64 格交付索引（2026-10-04）", "",
            "本索引读取既有尝试矩阵、同合同历史覆盖与当前报告；不创建数值尝试。JSON/CSV 各有 64 个唯一方法格，逐格列合同、当前原始 START/RESULT 层级、历史完整结果、数值完成与矩阵派生性质分类、独立证书资格，以及绝对宽度/图的原始来源。矩阵分类不是作者 checker 的逐字输出。",
            "", f"来源尝试数：{len(attempts)}。覆盖：" + "；".join(f"{k}={v}" for k, v in payload["coverage_counts"].items()) + "。",
            "", "[完整 JSON](archcomp26_delivery_index_20261004.json) · [64 行 CSV](archcomp26_delivery_index_20261004.csv)", "",
            "图或宽度文件可能采用该合同的另一已保存运行，不能把矩阵末次计时和首次范围拼成同一进程。历史全程不改名为本轮新运行。失败/Unknown/合同阻塞保留原样；没有独立端到端证书不抹掉已有数值记录，也不把数值完成提升为证明。", "",
            "| 实例 | P3 | Huan | Xiangru | Native | 图/宽度与缺口 |", "| --- | --- | --- | --- | --- | --- |"]
    short = {"current_numeric_full": "新数值完整", "historical_same_contract_numeric_full": "同合同历史完整",
             "no_full_numeric_evidence": "无完整/有具体停止", "contract_blocked": "合同阻塞"}
    for instance in IDS:
        group = [c for c in cells if c["instance_id"] == instance]
        note = "；".join(group[0]["missing_or_noncomparable"]) or "已列保存宽度和图；性质/证书限制见每格。"
        text.append("| " + " | ".join([instance] + [short[c["coverage_status"]] for c in group] + [note]) + " |")
    text += ["", "复建（只读源证据，输出此索引）：", "", "```bash", "python3 -B tools/build_archcomp26_delivery_index_nohash.py", "```", "",
             "验收：64 格唯一；当前选中尝试有原始结果镜像；8 格历史完整结果可读；图/宽度/合同路径存在；统计直接来自当前矩阵与覆盖附表。全部宽度以来源文档的 endpoint/tube、union/每盒 mean/max 对象为准，缺记录不估算。"]
    prefix.with_suffix(".md").write_text("\n".join(text) + "\n")
    print(json.dumps({"status": "indexed", "cells": 64, "coverage": payload["coverage_counts"]}))


if __name__ == "__main__":
    main()
