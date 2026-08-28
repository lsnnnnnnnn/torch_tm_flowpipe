# Torch–Flow* Brusselator C5 活跃 Range/Composition 因果闭包

日期：2026-08-28（UTC）

## 结论

本轮最终状态为：

```text
LIVE_RANGE_DOMINANT_CAUSE_NOT_IDENTIFIED__NO_C5
```

新鲜 Torch SR1000 C4 基线已经接受全部 1000 步并到达 `T=20`。因此 C4 没有被拒绝的终端尝试，C5 授权合同中强制要求的 terminal shadow causal gate 不存在且必须 fail closed。剩余的同对象 range/composition 差异虽然可以精确测量，但没有任何单一 operator 同时满足 80% gap elimination、至少三个后续 checkpoint 方向一致、下一步 margin 严格改善及被拒终端 margin 实质改善四个因果条件。本轮没有实现或声称任何 C5 求解器模式。

这也回答了本轮唯一科学问题：现有 range 差异不足以解释一个仍待消除的 C4 长前缀差距；C4 本身已闭合冻结的 `T=20` 请求，继续为局部数值 parity 修改 live solver 没有获得授权。

## 冻结合同与 provenance

冻结系统为：

```text
x' = 1 + x * (x * y - 4)
y' = x * (3 - x * y)
x(0) in [1.48, 1.52]
y(0) in [2.98, 3.02]
CPU float64, B1, order 6
h = 0.02, 1000 steps, T = 20
initial remainder = [-1e-4, 1e-4] per component
cutoff = 1e-10, validation_eps = 1e-12
endpoint repair/tightening = off
Torch SR capacity = Flow* SR capacity = 1000
```

提交身份严格分开：

- C4 scientific：`26323929d6f4fee0893478f6927ae76c5129bf47`
- C4 evidence/package：`89d0c17c3f6b3e99ac1d068f1573bda7e4f82cbe`
- C5 scientific：`null`，因为没有 operator 通过授权门
- 本轮 audit tooling 起点：`2d56fc423a63ef10cc6a24b4a43ba9055d2fe493`
- stock Flow*：`b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- 最终 package commit：由包含本文件与证据包的 Git commit 给出，并在最终 handoff 中报告；不能把 commit SHA 嵌入其自身内容

远端 `codex/vdp-terminal-sr1000-c4-closure-20260828` 的 tip 是完整 C4 evidence SHA，且 C4 scientific SHA 是其祖先。Phase 0 的 fresh-clone verifier 已确认两个对象均可由 GitHub remote 取得。

完整机器可读 provenance 见 `artifacts/runs/brusselator_live_range_c5_20260828/PROVENANCE.json`。

## C4 独立审计

C4 被归类为 **sound functional compatibility**，不是 bitwise 或 Flow* 源码逐行 parity。十二项审计全部通过：

1. 只在首次 raw self-map 成功后进入 refinement。
2. 首次 self-map 失败不能被 refinement 救活。
3. candidate polynomial 在 proposal 间固定且哈希不变。
4. 每个 proposal 都重新计算 remainder-dependent quantities。
5. generic C4 不使用未证明的 static cache。
6. subset decision 与 commit 都是 whole-vector atomic。
7. subset、nonfinite 或 evaluation failure 时保留最后完整 certified vector。
8. final decomposition、owner 与 generation 属于最后一次实际 commit。
9. validation epsilon、cutoff、roundoff 与 SR current owner 各有唯一 owner。
10. refined remainder 进入 SR queue 时不 double count。
11. rejection 不改变 queue、checkpoint、accepted boundary 或 current owner。
12. step 1 观察到的 8 次 replay 是达到 `stop_ratio` 前的结果，不是硬编码的 Flow* parity 声明。

固定 Flow* 源码的最大 refinement 宏为 490、replay 上限为 491，Torch 的 atomic update 更保守但更容易审计。详细逐项证据在 `C4_AUDIT.json`。

## 新鲜完整基线

| lane | accepted | horizon | rejected | solver seconds | queue reset | outcome |
|---|---:|---:|---:|---:|---:|---|
| stock Flow* | 1000 | 20.00 | 0 | 14.412807252 | n/a | completed |
| Torch SR1000 legacy raw compat | 357 | 7.14 | 1（attempt 358） | 700.069176808 | 0 | stopped |
| Torch SR1000 C4 refined raw compat | 1000 | 20.00 | 0 | 6083.447284393 | 1（完成 step 1000 后的容量事件） | completed |

所有 Torch certificate、owner accounting、sampling containment 与 rollback 检查均通过。legacy 在第一次 rejection 前没有 reset，因此原来的 `7.14 vs 20` 不能归因于 queue capacity/reset。C4 在 step 1000 后记录的 reset 不参与获得任何额外 acceptance。

C4/legacy 首个 published difference 位于 accepted step 1，八个 endpoint/tube bounds 都不同，最大绝对差为 `1.7548919124532603e-05`。首次 validation-margin difference 位于 attempt 2，limiting-margin delta 为 `4.6396362313595556e-10`。C4 最终 first-self-map limiting margin 为 `9.142451266096079e-05`，是正值且成功接受，不存在 rejected terminal attempt。

## Canonical exchange 与同对象矩阵

保存的 accepted checkpoints 为：

```text
1, 2, 3, 100, 200, 300, 996, 997, 998, 999, 1000
```

它们覆盖最初三步、首次 material/published divergence、100/200/300、C4 终点前五个 accepted boundaries 与最终 attempt。由于 C4 完成请求，无法也不应伪造一个 rejected terminal object。

每个 canonical object 包含完整 monomial 指数表、binary64 coefficient/endpoints hex、ordinary remainder、SR propagated history、current owner、normalization center/scale、domain、step power tables、cutoff 与源码哈希。11 个对象均可重建 post-right-map 和 post-queue，且 binary64 bitwise 相等。

同一对象的 286 行交换矩阵全部通过 exact/local outward oracle：

- A/B：Torch/Flow* polynomial-only endpoint range
- C/D：Torch/Flow* polynomial-only tube range
- E/F：加同一 ordinary remainder 后的 endpoint/tube range
- G/H：Torch normal pipeline 与 Flow* `polyRangeNormal + insert_ctrunc_normal + intEvalNormal`
- X1/X2：Torch live early endpoint cutoff 与 Flow* 实际 no-cutoff endpoint
- X3：post-scale cutoff diagnostic，仅用于 owner 审计，不能授权 live C5

Flow* harness 从固定提交的新临时 clone 构建。最小审计补丁只暴露 canonical importer 所需的 degree setter；它不修改 stock 数值路径。探针按 Flow* 自身 term ordering 排序后构造 polynomial，避免 importer 顺序成为伪差异。

## 固定顺序的 cause audit

材料性阈值为 `1e-12`。

| 顺序 | 比较 | 首次数值差 | 首次材料差 | 最大 bound delta | 判定 |
|---:|---|---|---|---:|---|
| 1 | cutoff ownership X1/X2 | step 1 endpoint，`1.7763568394002505e-15` | 无 | `3.552713678800501e-15` | 非材料性 |
| 2 | polyRangeNormal A/B | step 1 endpoint，`1.7763568394002505e-15` | 无 | `5.329070518200751e-15` | 非材料性 |
| 2 | polyRangeNormal C/D | step 1 tube，`3.9968028886505635e-15` | 无 | `2.1316282072803006e-14` | 非材料性 |
| 3 | insert/composition G/H | step 1 boundary，`2.42861286636753e-17` | step 100，`1.5557572070890036e-10` | `1.5713135480266427e-10` | 仅一个材料 checkpoint |
| 4 | normalization/right-map | 与 G/H 同一首次差异 | step 100 | `1.5713135480266427e-10` | 无独立持续因果链 |

首次可见 live decision/margin 差异出现在 boundary step 2 对 next attempt 3：H shadow 的 limiting-margin 改善只有 `5.55653613398821e-19`。在 boundary step 100 的改善为 `2.417770844642675e-15`；其余已注册点 `1, 3, 200, 300, 996, 997, 998, 999` 为零。10 个 shadow checkpoints 中仅 2 个方向性改善，不满足至少三个后续点方向一致，也没有达到材料性改善。

没有进入二元交互检查：当每个单项都无法通过不存在的 rejected-terminal gate 时，拼接多个小差异会违反预注册授权规则。

## C5 授权门

最终 gate 结果：

| gate | result |
|---|---|
| exact/local outward oracle | pass |
| owner/cache/atomicity audit | pass |
| frozen contract、queue policy、cutoff unchanged | pass |
| reporting-only exclusion | pass |
| first-material same-input gap elimination >= 80% | fail |
| 至少三个后续 checkpoints 方向一致 | fail |
| next-step limiting margin 严格、可归因改善 | fail |
| rejected-terminal margin 材料性改善 | fail closed：C4 无 rejection |

因此 `c5_authorization.json` 中 `authorized=false`、`c5_mode=null`、`c5_scientific_commit=null`。没有运行不适用的 C5 horizon、tightness 或三次 runtime production gate；三个 production matrices 明确把 `c5_applicable` 标为 false，没有用缺失数据伪造 C5 lane。

## 测试与复核

最终 targeted 命令覆盖 canonical round trip、endpoint/tube exact range、cutoff 与 composition owner、normalization、reporting/live 分离、same-input substitution、C5-off/C4 oracle、VDP C3、rollback、checkpoint/resume 与 tamper rejection：

```bash
conda run -n py11 pytest -q \
  tests/test_brusselator_c4_generic_refinement.py \
  tests/test_brusselator_c5_live_range.py \
  tests/test_brusselator_c5_evidence.py \
  tests/test_vdp_c2_post_accept_refinement.py \
  tests/test_accepted_boundary_sr.py \
  tests/test_vdp_c3_cross_step_queue.py
```

结果：`56 passed in 38.45s`。

全仓首次运行得到 `907 passed, 2 skipped, 1 failed`，唯一失败是历史脚本 `experiments/replay_fixed_support_fraction.py` 直接作为子进程启动时未加入仓库 `src/`。加入与其他 experiment entrypoints 相同的本地 source bootstrap 后，失败项单测为 `1 passed in 1.54s`。最终全仓重跑为 `908 passed, 2 skipped in 382.81s`，记录于 `raw/tests/full_final.log`。

此外，既有 C4 evidence verifier 与 VDP C3 native `T=10` verifier 均通过。最终包 verifier 同时校验完整 `SHA256SUMS` 覆盖、provenance、12 项 C4 audit、三条基线、canonical import/hash、286 行 exact matrix、terminal/no-C5 状态及 production matrix 维度。篡改测试证明即使重算被改文件的 SHA，伪造高状态仍会被语义规则拒绝。

## 复现

以下命令在仓库根目录运行；输出目录可替换为新的绝对路径：

```bash
conda run -n py11 python experiments/run_brusselator_sr1000_parity.py \
  --validation-mode flowstar_raw_remainder_compat \
  --lane-label torch_sr1000_legacy_fresh \
  --output-dir /absolute/path/legacy

conda run -n py11 python experiments/run_brusselator_sr1000_parity.py \
  --validation-mode flowstar_raw_remainder_compat_refined \
  --lane-label torch_sr1000_c4_full_prefix \
  --capture-c5-checkpoints \
  --output-dir /absolute/path/c4

conda run -n py11 python experiments/run_brusselator_second_system_flowstar.py \
  --source /srv/local/shengenli/flowstar_vdp_c3_baseline_b85a321 \
  --output-dir /absolute/path/flowstar

conda run -n py11 python experiments/export_brusselator_canonical_tm.py \
  --baseline-dir /absolute/path/c4 \
  --output-dir /absolute/path/canonical

conda run -n py11 python experiments/replay_brusselator_range_pipeline.py \
  --objects-dir /absolute/path/canonical \
  --flowstar-source /srv/local/shengenli/flowstar_vdp_c3_baseline_b85a321 \
  --output-dir /absolute/path/replay

conda run -n py11 python experiments/analyze_brusselator_live_range_cause.py \
  --c4-dir /absolute/path/c4 \
  --legacy-dir /absolute/path/legacy \
  --flowstar-dir /absolute/path/flowstar \
  --objects-dir /absolute/path/canonical \
  --replay-dir /absolute/path/replay \
  --output-dir /absolute/path/analysis

conda run -n py11 python scripts/verify_brusselator_c5_evidence.py
conda run -n py11 pytest -q
```

## Artifact map

证据包位于 `artifacts/runs/brusselator_live_range_c5_20260828/`：

- `PROVENANCE.json`：提交、冻结合同、fresh-run 与源码哈希
- `C4_AUDIT.json`：十二项独立审计
- `C4_FULL_PREFIX_BASELINE.json`：三条新鲜完整基线及终端测量
- `CANONICAL_OBJECT_SCHEMA.json` 与 `raw/canonical_exchange/`：规范对象和索引
- `same_object_range_matrix.csv` 与 `raw/range_replay/`：A–H/X1–X3 exact replay
- `first_live_range_divergence.json`：固定顺序的数值/material/live/next-step 因果链
- `terminal_shadow_replay.json` 与 `c5_authorization.json`：终端 replay 和逐门授权结果
- `production_matrix.csv`、`native_horizon_matrix.csv`、`runtime_matrix.csv`：完整 solver 测量
- `RESULT.json`：唯一规范状态
- `raw/c4_baseline/`、`raw/legacy_baseline/`、`raw/flowstar_baseline/`：命令、日志、segments、diagnostics 与 checkpoints
- `raw/tests/`：targeted、全套与既有 verifier 日志
- `SHA256SUMS`：除自身外所有包内文件的全覆盖哈希清单

失败或被后续复算取代的中间 harness/replay 目录没有进入最终包；最终包只包含使用最终源码哈希重新生成且通过 verifier 的对象与结果。
