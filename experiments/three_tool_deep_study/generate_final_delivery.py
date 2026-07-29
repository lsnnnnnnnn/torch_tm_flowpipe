#!/usr/bin/env python3
"""Generate repository-level reports and a complete curated-artifact manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoints() -> list[tuple[str, str]]:
    command = [
        "git",
        "log",
        "--reverse",
        "--format=%h%x09%s",
        "9024a8a..HEAD",
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        tuple(line.split("\t", 1))  # type: ignore[misc]
        for line in result.stdout.splitlines()
        if "\t" in line
    ]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _artifact_mapping(relative: str) -> tuple[str, str, str]:
    name = Path(relative).name
    plot_sources = {
        "01_one_step_tube_width_vs_h.png": (
            "one_step_summary.csv",
            "Protocol A: common one-step native tube",
        ),
        "02_one_step_endpoint_raw_width_vs_h.png": (
            "one_step_summary.csv",
            "Protocol A: common one-step raw endpoint",
        ),
        "03_exact_inflation_ratios.png": (
            "one_step_summary.csv",
            "Protocol A: analytic-reference inflation",
        ),
        "04_common_affine_carry_width_vs_time.png": (
            "affine_carry_summary.csv",
            "Protocol B: common affine carry",
        ),
        "05_common_box_carry_width_vs_time.png": (
            "box_carry_summary.csv",
            "Protocol C: common box carry",
        ),
        "06_affine_vs_box_carry.png": (
            "affine_carry_summary.csv + box_carry_summary.csv",
            "Protocols B/C: reset and dependency control",
        ),
        "07_native_low_order_width_curves.png": (
            "native_low_order_summary.csv",
            "Protocol D: accurately labelled native low order",
        ),
        "08_native_practical_width_runtime_pareto.png": (
            "native_pareto_summary.csv",
            "Protocol E: within-tool width/runtime Pareto",
        ),
        "09_successful_horizon_vs_runtime.png": (
            "native_pareto_summary.csv",
            "Protocol E: successful horizon/runtime",
        ),
        "10_polynomial_remainder_decomposition.png": (
            "component_ablation.csv",
            "Component attribution",
        ),
        "11_monomial_family_support.png": (
            "matched_basis_support.csv",
            "Matched-basis dictionary support",
        ),
        "12_torch_reset_order_ablation.png": (
            "component_ablation.csv",
            "Torch within-tool reset/order ablation",
        ),
        "13_diffreach_affine_quasi_symbolic_ablation.png": (
            "component_ablation.csv",
            "DiffReach within-tool basis/reset ablation",
        ),
        "14_flowstar_order_step_qr_symbolic_refinement_ablation.png": (
            "flowstar_component_ablation.csv",
            "Flow* within-tool component ablation",
        ),
        "15_matched_basis_results.png": (
            "matched_basis_summary.csv",
            "One-engine B1/B_DR/B2/B3 matched-basis experiment",
        ),
        "16_common_defect_vs_native_remainder.png": (
            "defect_summary.csv",
            "Common defect diagnostic versus native remainder",
        ),
        "17_runtime_decomposition.png": (
            "runtime_summary.csv",
            "Build/JIT/execution/memory decomposition",
        ),
        "18_failure_categories.png": (
            "failure_summary.csv",
            "Explicit failure classification",
        ),
    }
    if name in plot_sources:
        source, protocol = plot_sources[name]
        return source, "plot_results.py", protocol
    if name.startswith("controlled_"):
        return name, "run_controlled.py", "Protocols A/B/C"
    if name.startswith("native_"):
        producer = (
            "run_native.py"
            if name
            in {
                "native_torch.csv",
                "native_diffreach.csv",
                "native_flowstar.csv",
                "native_raw.csv",
                "native_low_order_summary.csv",
            }
            else "run_pareto.py"
        )
        return name, producer, "Protocols D/E"
    if name.startswith("pareto_repetitions_"):
        return name, "run_pareto.py", "Protocol E runtime repetitions"
    if name.startswith("acceleration_"):
        return name, "run_pareto.py", "Secondary same-tool backend throughput"
    if name.startswith("matched_basis_"):
        return name, "run_ablation.py", "Matched B1/B_DR/B2/B3 basis"
    if name in {"component_ablation.csv", "flowstar_component_ablation.csv"}:
        return name, "run_ablation.py", "Within-tool component attribution"
    if name == "defect_summary.csv":
        return name, "defect_diagnostic.py", "Common defect diagnostic"
    if name.startswith("flowstar_correctness"):
        return name, "flowstar_correctness.py", "Flow* correctness gate"
    if relative.startswith("flowstar_root_cause/"):
        return relative, "flowstar_correctness.py", "Flow* root-cause evidence"
    if relative.startswith("flowstar_adaptive_trajectory_audit/"):
        return (
            relative,
            "flowstar_adaptive_trajectory_audit.py",
            "Adaptive Flow* endpoint-path correctness evidence",
        )
    if name.startswith("bern_feasibility"):
        return name, "bern_feasibility.py", "RQ6 range-only feasibility"
    if name in {
        "one_step_summary.csv",
        "affine_carry_summary.csv",
        "box_carry_summary.csv",
        "raw_results.csv",
        "runtime_summary.csv",
        "failure_summary.csv",
    }:
        return name, "collect_results.py", "Normalized protocol summary"
    return name, "run_all.sh", "Run metadata or supporting evidence"


def _write_manifest(artifact: Path, destination: Path) -> int:
    selected = sorted(artifact.rglob("*.csv")) + sorted(
        (artifact / "plots").glob("*.png")
    )
    rows: list[dict[str, str]] = []
    for path in selected:
        relative = path.relative_to(artifact).as_posix()
        source, producer, protocol = _artifact_mapping(relative)
        rows.append(
            {
                "artifact_path": relative,
                "artifact_kind": (
                    "plot" if path.suffix == ".png" else "table"
                ),
                "authoritative_run": artifact.name,
                "protocol": protocol,
                "raw_source": source,
                "producing_script": producer,
                "semantic_notes": (
                    "sampling fields are deterministic non-proof sanity checks; "
                    "Pareto flags are within-tool/system/time only; unavailable "
                    "values are not encoded as zero"
                ),
                "sha256": _sha256(path),
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def generate(artifact: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    final = _json(artifact / "final_acceptance.json")
    correctness = _json(artifact / "correctness_checks.json")
    quality = _json(artifact / "artifact_quality_audit.json")
    environment = _json(artifact / "environment.json")
    bern = _json(artifact / "bern_feasibility.json")
    completion = _json(artifact / "RUN_COMPLETE")
    pareto = _csv(artifact / "native_pareto_summary.csv")
    failures = _csv(artifact / "failure_summary.csv")
    capabilities = _csv(artifact / "matched_basis_capabilities.csv")
    one_step = _csv(artifact / "one_step_summary.csv")
    affine = _csv(artifact / "affine_carry_summary.csv")
    box_carry = _csv(artifact / "box_carry_summary.csv")
    matched = _csv(artifact / "matched_basis_summary.csv")
    runtime = _csv(artifact / "runtime_summary.csv")
    canonical_report = (
        artifact / "three_tool_deep_study_report.md"
    ).read_text(encoding="utf-8")
    checkpoints = _checkpoints()
    repeated = [
        row
        for row in pareto
        if int(float(row.get("runtime_repetitions") or 0)) >= 10
    ]
    repositories = environment.get("repositories", {})
    adaptive = (
        correctness.get("flowstar", {})
        .get("adaptive_trajectory", {})
    )
    source_rel = Path("artifacts") / "authoritative" / artifact.name
    checkpoint_table = _table(
        ["SHA", "checkpoint"], [[sha, subject] for sha, subject in checkpoints]
    )
    repository_table = _table(
        ["repository", "SHA"],
        [
            [name, record.get("sha", "n/a")]
            for name, record in sorted(repositories.items())
        ],
    )
    capability_table = _table(
        ["tool", "basis", "status", "reason"],
        [
            [
                row.get("tool", ""),
                row.get("basis", ""),
                row.get("status", ""),
                row.get("reason", ""),
            ]
            for row in capabilities
        ],
    )
    header = f"""# Authoritative three-tool deep-study report

## Delivery status

This report describes the passed `{artifact.name}` artifact on branch
`codex/torch-flowstar-diffreach-deep-study`. It is authoritative for this
repository; the default branch may not yet include it.

- Final acceptance: **{final.get('passed', False)}**
- Ten-repetition gate required: **{final.get('require_ten_repetitions', False)}**
- Artifact-quality audit: **{quality.get('passed', False)}**
- Parsed CSV rows: **{quality.get('csv_total_rows', 0)}**
- Mandatory plots: **{final.get('mandatory_plot_count', 0)}**
- Repeated native configuration/system rows: **{len(repeated)}**
- Explicitly classified failure rows: **{len(failures)}**
- Curated artifact: `{source_rel.as_posix()}`

## What changed

The earlier literal “order-1 winner” claim is retracted. Torch, DiffReach, and
Flow* do not attach the same basis, validator, carry/reset contract, or
arithmetic meaning to the same order label. The valid study therefore separates
common raw one-step output, common affine carry, common box carry, accurately
labelled native modes, and native practical tradeoffs. It never compares a
legacy-tightened Torch endpoint against another tool's raw endpoint.

Flow*'s stock Riccati miss was traced to a variable-leaf truncation contribution
present in the full evaluator but absent from the cached remainder-only replay.
The record/replay patch and the independent full-Picard revalidation both
restore containment. The fixed-order-2, `h=0.05` Riccati stress point can
reject its configured candidate remainder; that is a configuration rejection,
not a crash or an overall Flow* failure. No experiment overwrites a Flow*
remainder after `advance`.

The earlier adaptive Van der Pol collapsed endpoint miss is also closed.  The
audit compares stock upstream, the original and identical generated harnesses,
the variable-leaf patch, and the adaptive full-Picard fallback.  It localizes
the miss to collapsed endpoint restriction: Flow*'s native composed flowpipe
evaluated on `tau=[h,h]` contains every deterministic sample.  The raw endpoint
now carries the explicit hull delta as independent remainder.  Repair passed:
**{adaptive.get('passed', False)}**; excluded from authoritative:
**{adaptive.get('repair', {}).get('excluded_from_authoritative', True)}**.

## Repository provenance

{repository_table}

## Pushed study checkpoints included before the full run

{checkpoint_table}

## Interpretation contract

Pareto dominance is computed only within one tool, system, and absolute
evaluation time. Common affine and box carry control the propagated
representation, not the native local construction. Box reset discards
correlation, although recentering can still reduce a later measured width, so a
ratio below one is not “negative dependency loss.” Deterministic trajectory
sampling is a bug-finding sanity check and never a proof of containment.

## Basis availability

{capability_table}

## BERN decision

BERN is a range-only feasibility component, not a fourth reachability solver.
Its clean-room float64 prototype contains all
{bern.get('cases', 0)} analytic cases and is stricter on
{bern.get('strictly_tighter_cases', 0)} cancellation cases. This supports
further work only after a formally enclosed sparse roundoff backend; it does
not provide integration, Picard validation, truncation handling, endpoint
substitution, reset, or plant/controller composition.

## Detailed generated study

"""
    (destination / "FINAL_REPORT.md").write_text(
        header + canonical_report, encoding="utf-8"
    )

    flow = correctness.get("flowstar", {})
    counts = flow.get("analytic_counts", {})
    parity = flow.get("original_parity", {})
    zh = f"""# 三工具 Taylor 模型深度研究：最终报告

## 交付状态

本报告对应通过验收的 `{artifact.name}` 运行，权威分支为
`codex/torch-flowstar-diffreach-deep-study`。默认分支可能尚未包含这些结果。
最终验收为 **{final.get('passed', False)}**，十次重复门槛为
**{final.get('require_ten_repetitions', False)}**，质量审计为
**{quality.get('passed', False)}**。质量审计解析了
{quality.get('csv_file_count', 0)} 个 CSV、共
{quality.get('csv_total_rows', 0)} 行；生成
{final.get('mandatory_plot_count', 0)} 张强制图。

## 核心结论与撤回

早期“同阶冠军”结论正式撤回。Torch 的 complete total-degree 基、
DiffReach 的 affine/restricted quasi-quadratic 基以及 Flow* 最低合法
二阶基并不相同；三者的验证器、余项语义、重置方式和数值后端也不同。
因此不能把三个工具标签相同或相近的 `order` 当成同一数学对象，也不能
给出跨工具的统一赢家。

有效比较被拆成五类：共同初始盒和步长下的原始单步结果、共同仿射 carry、
共同逐分量盒 carry、准确标注真实基的原生低阶配置，以及同一工具、同一
系统、同一绝对时刻内部的宽度/运行时间 Pareto 前沿。Torch 的
legacy-tightened 端点只保留为该工具内部诊断，绝不与其他工具的 raw
端点比较。没有达到请求终点的配置只报告成功时域，不能参与终点宽度比较。

## Flow* 正确性修复

Riccati 反例的根因不是精度不足，而是缓存路径不一致：完整表达式求值在
变量叶节点执行截断并产生区间贡献，旧的“只重放余项”缓存没有记录该项。
stock 结果因此漏掉解析真值。记录/重放补丁以及独立的 full-Picard
重新验证都恢复了解析包含；主修复行
{counts.get('primary_rows', 0)} 条，修复后解析违反
{counts.get('primary_analytic_violations', 'n/a')} 条，stock 反例保留
{counts.get('stock_analytic_violations', 'n/a')} 条。原始 Van der Pol
调度到 T=10 使用 {parity.get('original_segments', 'n/a')} 段，根因补丁
到 T=10 使用 {parity.get('root_cause_segments', 'n/a')} 段；补丁改变
自适应接受决策，所以不要求两者调度相同。固定二阶、`h=0.05` 的 Riccati
压力点可能拒绝其候选余项；这属于“配置被拒绝”，不是 Flow* 整体失败或
崩溃。实验也从不在 `advance` 后覆盖 Flow* 余项。

此前 Van der Pol 自适应路径的折叠端点漏包也已闭环。审计同时比较
upstream stock、原始/同构生成 harness、变量叶截断补丁和自适应
full-Picard fallback，定位到折叠端点限制路径。对同一已验证 flowpipe
直接在 `tau=[h,h]` 上做原生区间求值时，所有确定性样本均被包含；raw
端点现将两条路径的 hull 差值显式加入独立余项。修复通过：
**{adaptive.get('passed', False)}**；是否从权威结果排除：
**{adaptive.get('repair', {}).get('excluded_from_authoritative', True)}**。

## 共同协议与基

共同 CIR v2 明确记录状态顺序、时间变量、归一化域、中心、状态缩放、
稀疏指数、系数区间、独立/结构化余项、tube/endpoint、请求与接受步长、
请求与成功时域、验证状态、运行时间拆分和来源 SHA。缺失能力写成
`unavailable`，不会伪装成零。

{capability_table}

B1 是 complete affine；B_DR 是受限的局部时间/状态结构；B2 是 complete
quadratic；B3 只增加“一次局部时间乘二次状态依赖”，包括
`tau*xi_i*xi_j`，但排除一般三次状态项和 `tau^3`。匹配基实验在同一个
算术引擎、同一验证器和同一重置下比较这些字典，用于隔离“基”的作用，
而不是伪造三个原生工具都支持同一个字典。

## 正确性、运行时间与有效性

本次运行完成 {correctness.get('analytic_checks', 0)} 个解析检查、
{correctness.get('common_segment_point_checks', 0)} 个 CIR 点包含检查、
{correctness.get('common_segment_native_round_trip_checks', 0)} 个
原生/CIR 往返检查，以及
{correctness.get('native_validation_checks', 0)} 个原生验证检查。
确定性轨迹采样检查了
{correctness.get('trajectory_sanity', {}).get('checked', 0)} 个候选，
它只用于发现错误，不是证明。任何采样失败的候选都从主要数值 Pareto
集合中排除，但仍保留成功时域和失败分类证据。

运行时间把编译/JIT、稳态完整配置执行、重复次数和内存分开。JAX 融合与
JIT、C++ 编译/进程启动属于后端因素；多项式项数、Picard/细化轮数、
区间求值、符号窗口和重置属于算法工作量。不存在跨工具 Pareto 排名，
也不把不同绝对时刻的宽度/时间点互相支配。

## BERN/IBF 可行性

BERN 只作为多项式范围查询候选，不是第四个可达性工具。clean-room CPU
float64 原型对 {bern.get('cases', 0)} 个解析案例都包含精确范围，并在
{bern.get('strictly_tighter_cases', 0)} 个消去型案例上更紧。当前证据是
“精确代数 Bernstein 包络 + 保守浮点余量候选”，不是完整舍入证明。
在继续之前需要稀疏维数限制和形式化向外舍入后端。它本身不提供局部时间
积分、截断余项、Picard 验证、端点代入或多步重置；本研究也没有神经网络
控制器，因此不能据此推出 CROWN/β-CROWN 或 NN 抽象的直接收益。

## 局限与后续

有效结论限于上述共同协议、准确标注的原生配置以及工具内部消融。无效结论
包括“同阶全局赢家”、不同绝对时刻的宽度排名、以及把 MPFR 证明强度与
float64 候选等同。建议 Torch 后续支持规范化 affine/QR 重置、受限
time-state 基、改进多项式范围界和 overflow 归因、暴露验证器时间，并在
任何证明级声明之前加入严格 directed-rounding/MPFR 后端。

完整英文数值表、十一项逐题回答和全部有效性限制见 `FINAL_REPORT.md`；
权威表格和图片见 `{source_rel.as_posix()}`。
"""
    (destination / "FINAL_REPORT_ZH.md").write_text(zh, encoding="utf-8")

    frontier = [
        row
        for row in pareto
        if row.get("width_runtime_pareto", "").lower() == "true"
        and row.get("primary_numerical_eligible", "true").lower() == "true"
    ]
    frontier_table = _table(
        [
            "tool",
            "variant",
            "system",
            "time",
            "width",
            "successful horizon",
            "steady s",
            "memory KiB",
        ],
        [
            [
                row.get("tool", ""),
                row.get("variant", ""),
                row.get("system", ""),
                row.get("evaluation_time", ""),
                row.get("width_at_evaluation_time", ""),
                row.get("successful_horizon", ""),
                row.get("steady_full_configuration_time_s", ""),
                row.get("memory_kib", ""),
            ]
            for row in frontier
        ],
    )
    first_failure = adaptive.get("first_failure", {})
    conclusions = f"""# Final conclusions

## Authority

These conclusions are generated from the accepted `{artifact.name}` run on
`codex/torch-flowstar-diffreach-deep-study`. Acceptance and the recursive
artifact-quality audit both passed. The complete isolated pytest matrix
reported {completion.get('complete_pytest_totals', {}).get('passed', 0)}
passed, {completion.get('complete_pytest_totals', {}).get('skipped', 0)}
skipped, and {completion.get('complete_pytest_totals', {}).get('failed', 0)}
failed tests.

## Revoked conclusions

The earlier “same-order winner” and any Torch-tightened-versus-other-raw
ranking are revoked. Equal order labels do not denote equal polynomial
dictionaries, validators, reset contracts, or arithmetic. A failed prefix
cannot be ranked at another solver's requested final time, common-box carry is
a reset/control protocol rather than a native-solver ranking, and sampling is
not a proof.

## Flow* correctness findings

The Riccati stock miss was caused by a variable-leaf truncation interval that
the full evaluator produced but the cached remainder-only replay omitted. The
record/replay correction and an independent full-Picard revalidation both
restore analytic containment: the primary audit contains
{counts.get('primary_rows', 0)} rows with
{counts.get('primary_analytic_violations', 'n/a')} analytic violations and
{counts.get('primary_endpoint_tube_violations', 'n/a')} endpoint/tube
violations. The stock miss remains in the evidence as a regression target.

The adaptive Van der Pol miss belongs to the collapsed endpoint
restriction/evaluation path, not to the ODE/reference mapping, the
variable-leaf patch, adaptive full-Picard acceptance, or the verified native
flowpipe. The first discrepancy is segment
{first_failure.get('segment_index', 'n/a')}, state
{first_failure.get('state_index', 'n/a')}, at absolute time
{first_failure.get('absolute_time', 'n/a')}; the collapsed lower endpoint
{first_failure.get('flowstar_lower', 'n/a')} missed the DOP853 sanity sample
{first_failure.get('reference_lower', 'n/a')} by
{first_failure.get('lower_under_enclosure_gap', 'n/a')}. The native composed
flowpipe evaluated on fixed local time enclosed all tested samples. The
exporter therefore uses the hull of the collapsed and fixed-domain native
evaluations and places the hull delta in the independent remainder. The
repaired authoritative path has zero deterministic trajectory misses and is
not excluded. The original upstream/generated schedule parity remains
{parity.get('schedule_agreement', False)} with
{parity.get('original_segments', 'n/a')} segments to T=10.

## Comparability and trusted numerical results

- The {len(one_step)} one-step rows use matched ODEs, initial boxes, state
  order, steps, and raw endpoint/tube semantics. They support local enclosure
  observations, but not a cross-tool winner because native bases and
  validators remain different.
- The {len(affine)} common-affine and {len(box_carry)} common-box rows control
  the propagated representation. Only rows at the same requested absolute
  horizon are juxtaposed. Box/affine ratios measure the effect of this reset
  control; recentering can make a ratio below one, so it is not “negative
  dependency loss.”
- The {len(matched)} matched-basis rows compare B1, B_DR, B2, and B3 inside one
  arithmetic engine with one validator/reset contract. They isolate retained
  monomial families; they do not pretend all three native tools implement
  those dictionaries.
- Native-practical width/runtime dominance is valid only within one tool,
  system, and absolute time. The authoritative nondominated rows are:

{frontier_table}

All exact widths and horizons are in
`{source_rel.as_posix()}/one_step_summary.csv`,
`affine_carry_summary.csv`, `box_carry_summary.csv`,
`native_low_order_summary.csv`, and `native_pareto_summary.csv`. Runtime,
warm-up/build/JIT, ten repetitions, and memory are kept separate in
`runtime_summary.csv` ({len(runtime)} rows); no unavailable memory or
capability value is replaced by zero. Dependency/reset evidence is in the two
carry tables and `component_ablation.csv`. The associated figures are
`plots/01_*.png` through `plots/18_*.png`.

## Evidence strength

Riccati containment and small polynomial identities have analytic checks.
Flow* acceptance uses its native interval/Picard certificates, and CIR
round-trip checks establish export consistency. Torch and DiffReach preserve
their native validation statuses, but their float64 paths are not promoted to
MPFR-style formal roundoff proofs. DOP853 trajectory checks are deterministic
bug-finding sanity checks only; their zero-failure result is an admission gate,
not a certificate.

## BERN and remaining gaps

BERN is feasible only as a polynomial range backend: all
{bern.get('cases', 0)} analytic cases were contained and
{bern.get('strictly_tighter_cases', 0)} cancellation cases were tighter. It is
not a fourth reachability solver and presently lacks integration, Picard
validation, truncation accounting, endpoint substitution, reset, and a
formally outward-rounded sparse backend.

No unexplained native trajectory failure remains in an authoritative row.
Capability gaps remain explicit in `matched_basis_capabilities.csv`, including
unavailable exact-basis mappings and structured-remainder observability.
Further proof-grade claims for float64 Torch/DiffReach or the BERN prototype
require directed-rounding/MPFR evidence. The requested course PDFs were absent
from the server-wide filename audit, so `MATERIALS_MISSING.md` records all 16
exact names and `LITERATURE_MAP.md` does not claim page-level review.
"""
    (destination / "FINAL_CONCLUSIONS.md").write_text(
        conclusions, encoding="utf-8"
    )

    index = f"""# Artifact index

## Authoritative

- `{source_rel.as_posix()}` — passed ten-repetition run, quality-audited and
  curated; this is the sole authoritative numerical artifact for the deep
  study.
- `FINAL_REPORT.md` / `FINAL_REPORT_ZH.md` — detailed English and Chinese
  reports.
- `FINAL_CONCLUSIONS.md` — artifact-derived final conclusions and evidence
  boundary; replaces the superseded smoke placeholder.
- `RESULTS_MANIFEST.csv` — every curated CSV and plot mapped to its source,
  producer, protocol, and SHA-256.

## Correctness provenance retained

- `experiments/three_way_comparison_repair/results/20260728T140456Z` — repaired
  historical comparison delivery; superseded by this deep study for broad
  conclusions, retained for correction provenance.
- `experiments/three_tool_deep_study/flowstar_patches/fa39f7a_series` —
  authoritative Flow* patch series because the upstream remote was not
  writable from this environment.

## Invalidated or superseded, never deleted

- `experiments/three_way_common_contract/results/20260724T132534Z` — invalid:
  post-`advance` Flow* remainder overwrite and Torch tightened/raw mismatch.
- `experiments/first_order_three_way/results/20260723T173852Z` — superseded:
  unmatched bases; no cross-tool winner may be inferred.
- Earlier top-level Flow* patch files — superseded by `fa39f7a_series`.
- Ignored interrupted diagnostic runs `20260729T041318Z`,
  `20260729T041345Z`, `20260729T041354Z`, `20260729T041924Z`, and
  `20260729T053811Z` — recovery evidence only, not authoritative.
- Legacy `outputs/` diagnostics — historical implementation diagnostics, not
  three-tool parity evidence.
"""
    (destination / "ARTIFACT_INDEX.md").write_text(index, encoding="utf-8")

    reproduction = f"""# Reproducibility

## Frozen repositories

{repository_table}

The authoritative branch is `codex/torch-flowstar-diffreach-deep-study`.
Correctness-delivery base: `9024a8a29bdc0ad668a7c0620bd53872f4313cc8`.
The primary study uses CPU float64. CUDA availability was
`{environment.get('torch_probe', {}).get('cuda_available', 'n/a')}` and the
DiffReach devices were
`{environment.get('jax_probe', {}).get('devices', 'n/a')}`. Accelerator rows
are secondary implementation-throughput observations only.

## Environment and complete tests

```bash
cd {REPO}
/srv/local/shengenli/miniforge3/condabin/conda run -n py11 \\
  python -m pip install -e ".[test]"
DEEP_STUDY_RESULTS_DIR={artifact} scripts/run_complete_pytest.sh
```

Historical experiment directories contain repeated top-level module names,
so `run_complete_pytest.sh` uses isolated pytest processes while still
collecting every test file.

## Smoke, full run, audit, curation, and reports

```bash
experiments/three_tool_deep_study/run_smoke.sh
experiments/three_tool_deep_study/run_all.sh \\
  experiments/three_tool_deep_study/results/{artifact.name}
conda run -n py11 python \\
  experiments/three_tool_deep_study/audit_results.py \\
  --output-dir experiments/three_tool_deep_study/results/{artifact.name}
conda run -n py11 python \\
  experiments/three_tool_deep_study/curate_artifacts.py \\
  --source experiments/three_tool_deep_study/results/{artifact.name}
conda run -n py11 python \\
  experiments/three_tool_deep_study/generate_final_delivery.py \\
  --artifact-dir {source_rel.as_posix()}
```

The full run writes `RUN_COMPLETE` only after correctness, CIR, analytic,
parity, ten-repetition, plot, and artifact-quality gates pass. Curation refuses
an incomplete, failed, non-ten-repetition, or previously populated target.
`SHA256SUMS.csv` authenticates the curated bundle.

## Expected authoritative counts

- CSV files parsed: {quality.get('csv_file_count', 0)}
- CSV rows parsed: {quality.get('csv_total_rows', 0)}
- JSON files parsed: {quality.get('json_file_count', 0)}
- analytic checks: {correctness.get('analytic_checks', 0)}
- CIR point checks: {correctness.get('common_segment_point_checks', 0)}
- native/CIR round trips:
  {correctness.get('common_segment_native_round_trip_checks', 0)}
- mandatory plots: {final.get('mandatory_plot_count', 0)}
- BERN analytic cases: {bern.get('cases', 0)}

## Interpretation requirements

Use raw endpoints for cross-tool protocol tables. Treat all sampling as
non-proof. Compare Pareto points only within a tool/system/absolute-time group.
Never reinterpret a fixed-order configuration rejection as a crash, and never
mutate Flow*'s returned remainder after `advance`.
"""
    (destination / "REPRODUCIBILITY.md").write_text(
        reproduction, encoding="utf-8"
    )
    manifest_rows = _write_manifest(
        artifact, destination / "RESULTS_MANIFEST.csv"
    )
    return {
        "artifact": str(artifact),
        "destination": str(destination),
        "manifest_rows": manifest_rows,
        "checkpoint_count": len(checkpoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--destination", default=str(HERE))
    args = parser.parse_args()
    result = generate(
        Path(args.artifact_dir).resolve(),
        Path(args.destination).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
