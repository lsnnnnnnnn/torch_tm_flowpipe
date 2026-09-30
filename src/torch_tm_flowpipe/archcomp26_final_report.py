"""Render the evidence-only ARCH-COMP26 report draft or gated final report."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
from statistics import median
import sys
from typing import Any, Mapping

from .archcomp26_preflight import TERMINAL_STATUSES, resolve_cell, validate_matrix


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("benchmarks/archcomp26/manifest.json")
METHOD_LABELS = {
    "pytorch_gpu": "PyTorch/GPU",
    "huan": "Huan",
    "xiangru": "Xiangru",
    "flowstar_native": "Flow* native",
}
TIMING_FIELDS = (
    "process_total",
    "driver_total",
    "solver_core",
    "compile",
    "validation",
    "observer_output",
    "plot_report",
)
WIDTH_VIEWS = ("endpoint", "last_segment_tube", "full_horizon_tube")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _linked_record(root: Path, link: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_path = link.get("path")
    if raw_path is None:
        return None
    # validate_matrix has already checked repository containment, file bytes, and SHA.
    return _load((root / str(raw_path)).resolve())


def _fmt_number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.12g}"


def _fmt_extent(value: Mapping[str, Any]) -> str:
    return f"{value['kind']}={_fmt_number(value['value'])}"


def _fmt_domain(value: Mapping[str, Any]) -> str:
    return (
        f"{value['kind']}=[{_fmt_number(value['start'])},"
        f" {_fmt_number(value['end'])}]"
    )


def _md_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _phase_median(samples: list[Mapping[str, Any]], field: str) -> float | None:
    values = [sample["timing_s"][field] for sample in samples]
    if not values or any(value is None for value in values):
        return None
    return float(median(float(value) for value in values))


def timing_summary(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return full-horizon timing only; cold samples never enter steady statistics."""
    if record["run"]["status"] != "completed" \
            or not record["eligibility"]["performance_measurement_eligible"]:
        return None
    requested = record["run"]["requested_extent"]
    samples = [
        sample for sample in record["samples"]
        if sample["outcome"] == "completed"
        and sample["validated_extent"] == requested
    ]
    cold = [sample for sample in samples if sample["role"] == "cold"]
    steady = [sample for sample in samples if sample["role"] == "steady"]
    if not steady:
        return None
    process = [float(sample["timing_s"]["process_total"]) for sample in steady]
    return {
        "cold_n": len(cold),
        "cold_process_s": _phase_median(cold, "process_total"),
        "steady_n": len(steady),
        "steady_process_median_s": float(median(process)),
        "steady_process_min_s": min(process),
        "steady_process_max_s": max(process),
        "steady_phase_medians_s": {
            field: _phase_median(steady, field) for field in TIMING_FIELDS
        },
    }


def width_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return absolute widths with their explicit domains; never derive ratios."""
    if record["run"]["status"] != "completed" \
            or record["widths"]["status"] != "complete":
        return []
    rows: list[dict[str, Any]] = []
    for view in WIDTH_VIEWS:
        measurement = record["widths"][view]
        for coordinate in measurement["per_coordinate"]:
            rows.append({
                "view": view,
                "domain": measurement["domain"],
                "coordinate": coordinate["coordinate"],
                "lo": coordinate["union"]["lo"],
                "hi": coordinate["union"]["hi"],
                "width": coordinate["union"]["width"],
                "partition_mean": coordinate["per_partition_width"]["mean"],
                "partition_max": coordinate["per_partition_width"]["max"],
            })
    return rows


def final_readiness_reasons(
    manifest: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if manifest["execution_policy"]["experiments_paused"] is not False:
        reasons.append("experiments_paused")
    unresolved = [
        row["id"] for row in manifest["instances"]
        if row["contract"].get("status") != "resolved"
    ]
    if unresolved:
        reasons.append(f"unresolved_contracts={len(unresolved)}")
    if matrix.get("status") != "terminal":
        reasons.append(f"matrix_not_terminal={matrix.get('status')}")

    unassessed = 0
    nonterminal = 0
    support_mismatch = 0
    unsupported_without_blocker = 0
    for instance in manifest["instances"]:
        for method in manifest["methods"]:
            cell = resolve_cell(matrix, instance["id"], method)
            support = cell["support"]["status"]
            status = cell["run"]["status"]
            if support == "unassessed":
                unassessed += 1
            if status not in TERMINAL_STATUSES:
                nonterminal += 1
            if (support == "supported" and status == "skipped") or (
                support == "unsupported" and status != "skipped"
            ):
                support_mismatch += 1
            if support == "unsupported" and not cell["support"]["blockers"]:
                unsupported_without_blocker += 1
    if unassessed:
        reasons.append(f"unassessed_support={unassessed}")
    if nonterminal:
        reasons.append(f"nonterminal_cells={nonterminal}")
    if support_mismatch:
        reasons.append(f"support_run_mismatch={support_mismatch}")
    if unsupported_without_blocker:
        reasons.append(
            f"unsupported_without_evidence_blocker={unsupported_without_blocker}"
        )
    return reasons


def collect_report(root: Path = ROOT) -> dict[str, Any]:
    manifest = _load(root / MANIFEST_PATH)
    matrix_path = Path(manifest["execution_matrix"]["path"])
    matrix = _load(root / matrix_path)
    errors = validate_matrix(manifest, matrix, root=root)
    if errors:
        raise ValueError("invalid execution matrix: " + "; ".join(errors))

    rows: list[dict[str, Any]] = []
    run_counts: Counter[str] = Counter()
    for instance in manifest["instances"]:
        contract_record = None
        if instance["contract"].get("status") == "resolved":
            contract_record = _linked_record(root, instance["contract"]["record"])
        cells: dict[str, Any] = {}
        for method in manifest["methods"]:
            cell = resolve_cell(matrix, instance["id"], method)
            run_counts[cell["run"]["status"]] += 1
            cells[method] = {
                "cell": cell,
                "result": _linked_record(root, cell["result_record"]),
            }
        rows.append({
            "instance": instance,
            "contract_record": contract_record,
            "cells": cells,
        })
    return {
        "manifest": manifest,
        "matrix": matrix,
        "rows": rows,
        "run_counts": dict(sorted(run_counts.items())),
        "final_readiness_reasons": final_readiness_reasons(manifest, matrix),
    }


def _contract_lines(row: Mapping[str, Any]) -> list[str]:
    instance = row["instance"]
    record = row["contract_record"]
    if record is None:
        return [
            "- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。",
            f"- 待解决字段配置：`{instance['contract']['unresolved_field_profile']}`。",
            f"- 计划可视化：{instance['visualization']}。",
        ]
    fields = record["fields"]
    model = fields.get("dynamics", fields.get("transition", {}))
    model_semantics = {
        name: value for name, value in model.items()
        if name not in {"source", "sha256"}
    }
    horizon = fields.get("integration", {}).get("horizon")
    if horizon is None:
        horizon = fields.get("discrete", {}).get("transition_count")
    controller_update = fields["controller_update"]
    period = controller_update.get(
        "period", controller_update.get("period_steps")
    )
    property_value = fields["property"]
    property_semantics = property_value.get(
        "time_semantics", property_value.get("step_semantics")
    )
    record_link = instance["contract"]["record"]
    return [
        f"- 执行合同：`resolved`；记录 `{record_link['path']}`；"
        f"SHA-256 `{record_link['sha256']}`；profile `{record['profile']}`。",
        f"- 模型/转移：`{model.get('source')}`；SHA-256 "
        f"`{model.get('sha256')}`；语义 `{_json_cell(model_semantics)}`。",
        f"- 控制器：`{fields['controller']['source']}`；SHA-256 "
        f"`{fields['controller']['sha256']}`；I/O 顺序 "
        f"`{_json_cell(fields['controller']['input_output_order'])}`。",
        f"- 变量顺序：`{json.dumps(fields['variable_order'], ensure_ascii=False)}`。",
        f"- 初始集：`{fields['initial_set']['source']}`；SHA-256 "
        f"`{fields['initial_set']['sha256']}`；分区 "
        f"`{_json_cell(fields['initial_set']['partitions'])}`；boxes SHA-256 "
        f"`{fields['initial_set']['boxes_sha256']}`。",
        f"- 扰动：`{_json_cell(fields['disturbance'])}`。",
        f"- 请求时域/步数：`{horizon}`；性质："
        f"`{property_value['formula']}`；时间/步语义："
        f"`{_md_cell(property_semantics)}`；通过条件："
        f"`{property_value['pass_condition']}`。",
        f"- 逻辑控制日程：周期 `{period}`；更新次数 "
        f"`{controller_update['scheduled_updates']}`；"
        f"`{controller_update['schedule_semantics']}`。",
        f"- 计划可视化：{instance['visualization']}。",
    ]


def _command_text(cell: Mapping[str, Any]) -> str:
    argv = cell["command"]["argv"]
    return "—" if argv is None else f"`{_md_cell(shlex.join(argv))}`"


def _tagged_text(value: Mapping[str, Any]) -> str:
    mode = value.get("mode", "unresolved")
    raw = value.get("value")
    return str(mode) if raw is None else f"{mode}:{_fmt_number(raw)}"


def _json_cell(value: Any) -> str:
    return _md_cell(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))


def _method_configuration(cell: Mapping[str, Any]) -> dict[str, str]:
    integration = cell["numerics"]["integration"]
    remainder = cell["numerics"]["remainder"]
    controller = cell["controller_execution"]
    checker = cell["property_checker"]
    return {
        "integration": "; ".join((
            f"h={_tagged_text(integration['step_size'])}",
            f"work={_tagged_text(integration['solution_order'])}",
            f"point={_tagged_text(integration['point_order'])}",
            f"validation={_tagged_text(integration['validation_order'])}",
            f"semantics={integration['semantics'] or '—'}",
        )),
        "remainder": "; ".join((
            f"cutoff={_tagged_text(remainder['cutoff'])}",
            f"cap={_tagged_text(remainder['cap'])}",
            f"SR={_tagged_text(remainder['symbolic_queue'])}",
            f"semantics={remainder['semantics'] or '—'}",
        )),
        "controller": "; ".join((
            f"updates={controller['scheduled_updates'] if controller['scheduled_updates'] is not None else '—'}",
            f"NN={_tagged_text(controller['nn_calls'])}",
            f"semantics={controller['nn_call_semantics'] or '—'}",
        )),
        "checker": "; ".join((
            f"mode={checker['mode']}",
            f"id={checker['identity'] or '—'}",
            f"early-stop={checker['early_stop_policy'] or '—'}",
            f"semantics={checker['semantics'] or '—'}",
            f"certificate={checker['certificate_semantics'] or '—'}",
        )),
    }


def render_markdown(report: Mapping[str, Any], *, final: bool = False) -> str:
    blockers = list(report["final_readiness_reasons"])
    if final and blockers:
        raise ValueError("final report is not ready: " + ", ".join(blockers))
    manifest = report["manifest"]
    methods = list(manifest["methods"])
    title_suffix = "最终版" if final else "草稿（不可作为最终成绩）"
    lines = [
        f"# ARCH-COMP26 非 VCAS 四方完整实验报告：{title_suffix}",
        "",
        "> 本文只使用当前 2026 manifest、哈希绑定实例合同和哈希绑定结果记录。",
        "> 历史 Huan、旧 PyTorch 与旧 Flow* 数字不进入本报告的当前四方结果表。",
        "",
        "## 摘要与边界",
        "",
        "目标是在同一冻结 benchmark 合同下比较 PyTorch/GPU、Huan、Xiangru "
        "和 Flow* native 的完整性、进程时间与绝对 flowpipe 宽度。共享驱动仅用于"
        "控制变量，不等同于三套独立 NNCS 产品。",
        "",
        f"- 报告状态：**{'final-ready' if not blockers else 'not final-ready'}**。",
        f"- 实验暂停：**{'是' if manifest['execution_policy']['experiments_paused'] else '否'}**。",
        f"- 覆盖：{len(report['rows'])} 个实例 × {len(methods)} 个方法 = "
        f"{len(report['rows']) * len(methods)} 个 cell。",
        "- 当前 run 计数：`" + ", ".join(
            f"{key}={value}" for key, value in sorted(report["run_counts"].items())
        ) + "`。",
        "- 最终发布门：`" + (", ".join(blockers) if blockers else "passed") + "`。",
        "",
        "## 比较规则",
        "",
        "- 正式目标为 1 次冷启动和 5 次独立进程 steady；冷启动不进入 steady 中位数。"
        "长任务可预先声明较少 steady 次数并写明原因，但不得据此取得稳定排名资格。",
        "- 只报告完整请求时域且明确允许性能测量的时间；失败前缀不外推完成时间。",
        "- 宽度始终给绝对上下界、union width 和每分区 mean/max；本报告不计算宽度比。",
        "- 不同共同前缀、domain、变量顺序或单位不会合并为同一比较域。",
        "- `failed`、`timeout`、`interrupted` 与有证据的 `unsupported/skipped` 都保留。",
        "",
        "## 全部非 VCAS 配置覆盖矩阵",
        "",
        "| 实例 | " + " | ".join(METHOD_LABELS[method] for method in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for row in report["rows"]:
        values = [
            f"`{row['cells'][method]['cell']['run']['status']}`"
            for method in methods
        ]
        lines.append(f"| `{row['instance']['id']}` | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## 旧 14 项到 2026 manifest 的差异索引",
        "",
        "旧结果仅作回归线索；下表不会把旧完成状态或时间提升为 2026 重跑成绩。",
        "",
        "| 2026 实例 | 旧配置候选 | 映射状态 | 合同状态 | 已知差异/阻断数 |",
        "|---|---|---|---|---:|",
    ])
    for row in report["rows"]:
        instance = row["instance"]
        legacy = instance["legacy_candidate"]
        lines.append(
            f"| `{instance['id']}` | `{legacy['config_id'] or '—'}` | "
            f"`{legacy['status']}` | `{instance['contract']['status']}` | "
            f"{len(instance['known_issues'])} |"
        )

    for index, row in enumerate(report["rows"], start=1):
        instance = row["instance"]
        lines.extend([
            "",
            f"## {index}. {instance['benchmark']} — {instance['instance']} "
            f"(`{instance['id']}`)",
            "",
            "### 模型、控制器、初始集合与性质",
            "",
            *_contract_lines(row),
            "",
            "### 完整配置、状态与复现入口",
            "",
            "| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | "
            "updates / NN | arithmetic | hardware / runtime | checker / early-stop | "
            "命令 / cwd | source / binary identity | 结果记录 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])
        for method in methods:
            cell = row["cells"][method]["cell"]
            config = _method_configuration(cell)
            source_binary = {
                "source": cell["source_identity"],
                "binary": cell["binary_identity"],
            }
            result_path = cell["result_record"]["path"] or "—"
            lines.append(
                f"| {METHOD_LABELS[method]} | `{cell['support']['status']}` / "
                f"`{cell['run']['status']}` | `{_md_cell(config['integration'])}` | "
                f"`{_md_cell(config['remainder'])}` | "
                f"`{_md_cell(config['controller'])}` | "
                f"`{_json_cell(cell['arithmetic'])}` | "
                f"`{_json_cell(cell['runtime'])}` | "
                f"`{_md_cell(config['checker'])}` | {_command_text(cell)} / "
                f"`{_md_cell(cell['command']['cwd'] or '—')}` | "
                f"`{_json_cell(source_binary)}` | "
                f"`{result_path}` |"
            )

        lines.extend([
            "",
            "### 完整性、性质与结果资格",
            "",
            "| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | "
            "性质 / 证书 | soundness / scope | formal / performance / ranking |",
            "|---|---|---|---:|---|---|---|",
        ])
        for method in methods:
            record = row["cells"][method]["result"]
            if record is None:
                lines.append(
                    f"| {METHOD_LABELS[method]} | — | — | — | — | — | 无结果记录 |"
                )
                continue
            run = record["run"]
            property_value = record["property"]
            eligibility = record["eligibility"]
            lines.append(
                f"| {METHOD_LABELS[method]} | "
                f"`{_fmt_extent(run['requested_extent'])}` / "
                f"`{_fmt_extent(run['validated_extent'])}` | "
                f"`{str(run['requested_horizon_completed']).lower()}` | "
                f"{run['accepted_steps']} / {run['rejected_steps']} / "
                f"{run['nn_calls']} | `{property_value['status']}` / "
                f"`{property_value['certificate_status']}` | "
                f"`{eligibility['numerical_soundness_class']}` / "
                f"`{eligibility['soundness_scope']}` | "
                f"`{str(eligibility['formal_claim_eligible']).lower()}` / "
                f"`{str(eligibility['performance_measurement_eligible']).lower()}` / "
                f"`{str(eligibility['cross_tool_ranking_eligible']).lower()}` |"
            )

        lines.extend([
            "",
            "### 时间",
            "",
            "| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | "
            "driver / solver / validation / observer / plot median (s) | 排名资格 |",
            "|---|---:|---:|---:|---:|---|",
        ])
        time_rows = 0
        for method in methods:
            record = row["cells"][method]["result"]
            summary = timing_summary(record) if record is not None else None
            if summary is None:
                continue
            phases = summary["steady_phase_medians_s"]
            lines.append(
                f"| {METHOD_LABELS[method]} | "
                f"{_fmt_number(summary['cold_process_s'])} | "
                f"{summary['steady_n']} | "
                f"{_fmt_number(summary['steady_process_median_s'])} / "
                f"{_fmt_number(summary['steady_process_min_s'])} / "
                f"{_fmt_number(summary['steady_process_max_s'])} | "
                f"{_fmt_number(phases['driver_total'])} / "
                f"{_fmt_number(phases['solver_core'])} / "
                f"{_fmt_number(phases['validation'])} / "
                f"{_fmt_number(phases['observer_output'])} / "
                f"{_fmt_number(phases['plot_report'])} | "
                f"`{str(record['eligibility']['cross_tool_ranking_eligible']).lower()}` |"
            )
            time_rows += 1
        if not time_rows:
            lines.append("| — | — | — | — | — | 当前无合格的完整时域时间样本 |")

        lines.extend([
            "",
            "### 绝对宽度与共同前缀",
            "",
            "| 方法 | view | domain | 坐标 | lo | hi | union width | "
            "partition mean | partition max | 排名资格 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        count = 0
        for method in methods:
            record = row["cells"][method]["result"]
            if record is None:
                continue
            for width in width_rows(record):
                lines.append(
                    f"| {METHOD_LABELS[method]} | `{width['view']}` | "
                    f"`{_fmt_domain(width['domain'])}` | "
                    f"`{width['coordinate']}` | {_fmt_number(width['lo'])} | "
                    f"{_fmt_number(width['hi'])} | {_fmt_number(width['width'])} | "
                    f"{_fmt_number(width['partition_mean'])} | "
                    f"{_fmt_number(width['partition_max'])} | "
                    f"`{str(record['eligibility']['cross_tool_ranking_eligible']).lower()}` |"
                )
                count += 1
        if not count:
            lines.append("| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |")

        unresolved = instance.get("known_issues", [])
        failure_rows = []
        plot_rows = []
        for method in methods:
            cell = row["cells"][method]["cell"]
            record = row["cells"][method]["result"]
            if record is not None:
                for artifact in record["artifacts"]:
                    role = artifact["role"]
                    if any(token in role.lower() for token in ("plot", "figure", "matlab")):
                        plot_rows.append(
                            f"- {METHOD_LABELS[method]} `{role}`："
                            f"`{artifact['path']}`；SHA-256 `{artifact['sha256']}`。"
                        )
                trajectory = record["widths"]["trajectory_artifact"]
                if trajectory is not None:
                    plot_rows.append(
                        f"- {METHOD_LABELS[method]} `trajectory`："
                        f"`{trajectory['path']}`；SHA-256 `{trajectory['sha256']}`。"
                    )
            if cell["run"]["status"] in {"failed", "timeout", "interrupted", "skipped"}:
                failure_rows.append(
                    f"- {METHOD_LABELS[method]}：`{cell['run']['status']}` / "
                    f"`{cell['run']['failure_category']}` — {cell['run']['failure_detail']}"
                )
        lines.extend([
            "",
            "### Flowpipe 图、失败与未决项",
            "",
            f"- 目标图：{instance['visualization']}；只接受结果记录中哈希绑定的图/轨迹。",
            *(plot_rows or ["- 当前无哈希绑定的图或轨迹记录。"]),
            *(f"- 未决：{issue}" for issue in unresolved),
            *failure_rows,
        ])
        if not unresolved and not failure_rows:
            lines.append("- 未记录失败或未决项。")

    lines.extend([
        "",
        "## Huan QUAD 原因分析与 strict/parity 边界",
        "",
        "原因、模式、移植技巧与历史证据见 "
        "[`docs/HUAN_QUAD_SPEED_AND_MODES.md`](HUAN_QUAD_SPEED_AND_MODES.md)。"
        "该文档不向本报告的 2026 结果表注入历史时间。",
        "",
        "## 绘图功能与 MATLAB 示例",
        "",
        "图层、tube/endpoint 语义与 MATLAB/替代渲染边界见 "
        "[`docs/flowpipe_plotting.md`](flowpipe_plotting.md) 和 "
        "[`docs/FLOWPIPE_PLOT_VALIDATION_20261001.md`](FLOWPIPE_PLOT_VALIDATION_20261001.md)。",
        "",
        "## 统一数据包与发布",
        "",
        "每个终态 cell 的 result record 是索引入口，必须绑定原始单次样本、失败记录、"
        "结果/日志/命令、轨迹与宽度 artifact。大张量不嵌入报告；其位置与 SHA 只从"
        "已验证记录读取。Markdown 通过最终门后，才可据同一数据生成 DOCX/PDF。",
        "",
        "## 最终发布门",
        "",
    ])
    if blockers:
        lines.append("当前阻断：")
        lines.extend(f"- `{reason}`" for reason in blockers)
    else:
        lines.append("- `passed`：所有合同与 64 个 cell 已终态并通过共享证据校验。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="require the final-ready gate")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write", type=Path)
    output.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    try:
        text = render_markdown(collect_report(), final=args.final)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    elif args.check:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != text:
            print(f"stale ARCH-COMP26 report: {args.check}")
            return 1
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
