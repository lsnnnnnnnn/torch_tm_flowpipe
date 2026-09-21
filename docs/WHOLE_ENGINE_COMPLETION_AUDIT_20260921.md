# 完成交付证据审计

首轮审计时间：2026-09-21 13:59 UTC；最终证据收尾复核：14:30 UTC。已完整读取 `GOAL_PYTORCH_GPU_MAINLINE.md`；本次只读检查已有小型原始摘要、receipt、测试日志/定义及实际入口源码，没有启动求解器、GPU任务、测试或重CPU工作。最终配对已完成，本次仅检查其本地验收文件、分析/复现产物、缓存记录和中文报告，没有重复采样。

`PROVED` 仅表示表中限定的工程事实已有直接证据，不表示全程序浮点 soundness 或形式证明。所查条目没有未满足的 `PENDING` 或 `MISSING` 工程/实验要求；最终性能验收和中文路线结论均已有对应证据。审计结论限定于本Goal的既定双系统、固定输入和已测工作量，不扩张成所有系统/批量的收益承诺。

路径简称：

- `R` = `/srv/local/shengenli/whole_engine_feasibility_20260921T120109Z`
- `E` = `R/engine_cutoff_aligned`，引擎 `280abb400610f56210a7a5be61d5f98be3e27251`
- `T` = `/srv/local/shengenli/whole_engine_final_warm_timing_20260921T134430Z`
- 本地结果目录 = `/Users/shengenli/Documents/ChatGPT/verification/results/whole_engine_feasibility_20260921`
- 支持入口及最终配对实现固定为 `445218b757ea4e8cd17a488ec0bff13529e59d6f`。最终文档交付HEAD由根代理完成提交后写入外部清单 `R/DELIVERY_STATE.json`；本文不写自身所属提交哈希，不将之后的文档提交冒充测量时源码。
- 最终小型证据包位于本地 `final_timing/`；已实际读取 `ACCEPTANCE.json`、`LOCAL_REPRODUCTION.json`、`runs/timing_analysis.json` 和两份 `cache_evidence_after_*.json`，并核验分析JSON/脚本/CSV哈希。

| GOAL要求 | 已实际读取的权威证据 | 状态与限定结论 |
|---|---|---|
| 三方来源、版本和隔离工作树 | `R/SESSION.json`；首轮直接 `git rev-parse HEAD / branch --show-current / status --porcelain`；最终 `final_timing/ACCEPTANCE.json:source_states_after` | **PROVED**。主用户源起点 `e4e8d95`、保存严格源 `743f620`；主工作目录保持独立codex分支；正式配对副本结束时仍为干净 `445218b`，独立 E 为干净 `280abb4`。Xiangru 保存快照实际为干净 `1c16d4e`。最终文档身份另见外部交付清单。 |
| 保护旧工作树，不覆盖旧资产 | 直接读取 `/srv/local/shengenli/torch_tm_flowpipe` 的 HEAD/branch/status；保存严格源同项检查；隔离根目录和分支身份 | **PROVED（隔离与现存状态）**。旧用户目录仍为 `26a254e`、`codex/flowstar-raw-remainder-compat`，原有四个修改文件及未跟踪 docs/worktrees 仍在；保存严格源仍干净 `743f620`。本次没有进行旧 dirty 文件逐字节历史鉴证；不把当前 status 当作这种证明。 |
| 冻结两系统、原盒、h/order/cutoff/余项/队列，不换分盒 | `repo/experiments/whole_engine_feasibility/baseline.py:frozen_case`；四份 `runs/cutoff_aligned_{cpu,gpu}_{van_der_pol,brusselator}/summary.json` 的 case/settings；连续门中的分盒信息 | **PROVED**。原盒明确 `original=true`；VDP h=.01/order4/SR100、Bruss h=.02/order6/SR1000，strict、cutoff1e-10、余项1e-4、refinement491。B2固定lane0/31，B32既有8×4计划，分盒哈希 `717897f2…`；输入用 Fraction 检查向外包含精确目标。 |
| 完整PyTorch/CUDA状态推进，候选独立连续运行 | `candidate.py:run/advance_transaction/save_factored`；上述实际GPU summary的 device/extensions/source | **PROVED**。状态与SR在候选张量路径中逐步推进；没有循环调用旧CPU求解器或周期导入参考。接受状态由候选自身保存；CPU承担控制、同步状态读取、可选导出和离线观察。 |
| 保存严格版必要数值修复；fresh误差跨步传递、不重复旧历史 | `E/tests/soundness/test_sr_fresh_error.py`；`runs/strict_carry_regression/{before.log,evidence.json,py11_final.log}`；`E/src/flowstar_gpu/{flowpipe.py,sparse_exec.py}` 的fresh追加位置 | **PROVED（具体修复/反例）**。旧三步真实shear欠包络有独立Fraction依据；fresh增量按归一化前坐标记入J，测试检查下一真实步包含、无逐步reset绕过以及不重复累计旧shear。旧98项回归日志保留。不能由此推论所有算术路径都已证明。 |
| strict cutoff服从配置、保留parity语义 | `runs/cutoff_alignment_regression/{before.log,final.log,evidence.json}`；同一 `test_sr_fresh_error.py` 的dense/sparse×strict/parity真实两步测试 | **PROVED**。strict两项先失败后修复，parity保留旧阈值；最终四文件回归 **102 passed、2 skipped、13 deselected**，CPU py11/torch2.5.1+cu121。日志与命令明确 `-m 'not cuda'`，没有声称这是全套CUDA测试。 |
| 实际CUDA链的已知carry反例验证 | `runs/gpu_carry_regression/{command_environment.json,receipt.json,stdout.jsonl}`；`runs/gpu_carry_regression_check.py` | **PROVED（六例）**。最终干净280引擎、GPU2、dense/sparse×三组半径，各真实三步全部通过，receipt退出0；日志每例记录真实cutoff触发。Fraction评估不调用生产范围helper，并核实非负仿射全域最大值前提。 |
| 复用已有局部算术依据，避免未调用helper冒充实际链 | 102项回归的四文件精确命令；`E/tests/soundness/test_strict_proof_contract.py` 的测试定义；最终生产settings及实际生命周期测试 | **PROVED（有限回归范围）**。现存测试明确覆盖SR矩阵队列/reset、保留系数精确有理包含、端点系数误差收费、FTZ启动拒绝、非有限值/溢出失败及实际self-map/refinement分派。此项只核对证据和生产路径绑定，不重新证明整数幂或所有CUDA原语。 |
| 拒绝不提交、不同lane隔离、reset后不丢范围 | `repo/experiments/whole_engine_feasibility/test_candidate.py`；`runs/cutoff_aligned_adapter_cpu_tests.log` | **PROVED（3项真实生命周期测试）**。一lane真实拒绝、另一lane真实接受时整个batch保留原状态及非空SR；B2与各B1独立推进比较；容量2的解析线性任务跨reset2/4仍包含独立Fraction真值。5项日志还含2项精确导出检查，全部OK。 |
| B1/B2短检查及两系统CUDA B2×120、VDP历史清空 | `runs/cutoff_aligned_continuity_gate.json` 的命令、receipt和details；支持入口CPU/CUDA smoke日志 | **PROVED**。两植物各 `[120,120]`、completed=true/failure=null；VDP在100步reset。每植物离线观察240模型/960行，门检查全部通过。短B1 CPU/CUDA smoke各两系统2步；真实B2短推进同时由上述生命周期定义/日志覆盖。 |
| 最终引擎两系统原始单盒B1×1000 | 实际读取 `runs/cutoff_aligned_cpu_van_der_pol/summary.json`、`cutoff_aligned_cpu_brusselator/summary.json`、`cutoff_aligned_gpu_van_der_pol/summary.json`、`cutoff_aligned_gpu_brusselator/summary.json` | **PROVED**。四份均 `[1000]`、completed=true、failure=null、干净280引擎；GPU用CUDA扩展。VDP重置100…1000，Bruss在1000。辅助长程耗时不冒充正式配对性能。 |
| 保留旧失败前缀，不能补齐不存在步骤 | `runs/horizon_{van_der_pol,brusselator}/summary.json` 与现存非空 `factored.jsonl.gz` | **PROVED**。旧cff版本真实974/715接受，975/716拒绝且 `batch_committed=false`；原始压缩模型仍在，和最终1000步版本分开。 |
| 完整Taylor模型、四通道共同observer，不丢区间系数 | `candidate.py:save_factored`；`export.py`相关测试定义；`observe.py`实际导入 `experiments.xiangru_adoption.common.measure`；`observe_saved.py`；最终两份 `*_observed/summary.json` | **PROVED**。每植物1000模型/4000行，complete_observation=true；完整代入保留全部次数和区间系数，binary64转换向外舍入。每步endpoint/tube×x/y均计数；观察耗时独立57.0199s/818.9144s。 |
| CPU/Flow*亦经共同observer；合法匹配时域 | `runs/reference_exact_observer/{cpu_vdp,cpu_brusselator,flowstar_vdp,flowstar_brusselator}/summary.json`；共同observer源及wrapper哈希 | **PROVED**。CPU/Flow*读取保存完整模型而非旧bounds；共同measure身份哈希一致。CPU完整匹配1000步；Flow*第1000步域缩短且标称endpoint域外，匹配仅999步，例外和尾步单列。 |
| 全程质量median/p95/max及位置、绝对边界差、近零规则 | `runs/quality_comparison_complete_metrics/{analyze.py,summary.json,comparison.csv}`，本地同名副本；先前已实际执行此有界离线脚本 | **PROVED**。16项比较/15992逐步行；宽度比及上下界绝对差均有median、线性插值p95、max+step+time/精确时间；signed delta在远程stepwise.csv。固定64×ulp尺度规则实际排除0条。六份来源哈希匹配；原16项median/max/step逐项复现。 |
| 同引擎CPU/CUDA设备收益与换引擎收益区分 | `runs/device_control_results.json` 的12份receipt及每组完成门、命令/环境/时间 | **PROVED（独立设备对照）**。每植物3对、每次640接受；core7/GPU2，来源锁280。VDP CPU更快；Bruss预热求解CPU/CUDA≈1.354。该证据不替代最终Gr配对，也不代表完整新进程延迟收益。 |
| Gr直接基线、最终版本每植物3组交替配对、固定资源和边界 | `T/PREREGISTRATION.json`、`T/launch.json`；本地 `final_timing/{ACCEPTANCE.json,LOCAL_REPRODUCTION.json,runs/timing_analysis.json}` 的每样本checks、pairs/aggregates；根代理另逐份核对服务器12份原始summary/receipt | **PROVED**。12/12正式样本退出0、每次640接受，共7680 lane-steps；每植物固定三对，顺序Gr→候选/候选→Gr/Gr→候选。所有来源/输入/设备/环境/快照检查通过，源445/引擎280、GPU3/core2、单线程；两次B32×2预热另存共128接受。全部单次和离散统计已报告；VDP/Bruss的solve逐对倍率中位数266.315×/343.358×，process为71.939×/96.325×。本地Python3.11复现pairs/aggregates及CSV全部一致；没有混入旧cff样本。 |
| 正式调用边界、预热/编译/导出单列、超时与资源指标 | `candidate.py`初始化/同步/计时/预算源码；`final_timing/runs/timing_analysis.json:{timer_definitions,startup_and_cache}`；`cache_evidence_after_warmups.json`、`cache_evidence_after_formal.json`；`ACCEPTANCE.json`；主REPORT资源字段及原始summary/观察summary | **PROVED**。solve含初始化、完整advance、SR拷贝/提交和同步；三个扩展预加载在solve外，模型导出/观察独立，process另含启动/加载/序列化。两次预热全部检查通过；三扩展的.so、.ninja_log、build.ninja共9个文件的SHA256/mtime_ns/size逐项不变，没有正式扩展重编译。候选启动0.628–0.637s及预算/分配峰值/RSS口径已披露。验收记录无在运行的自有benchmark进程，GPU2/GPU3均0MiB。共享主机非独占，起止快照不证明连续独占。 |
| Flow*有限短对照、沿用现有驱动且不混为同算法设备倍率 | `runs/flowstar_b32x20_{van_der_pol,brusselator}/baseline.json`；`baseline.py:run_flowstar` | **PROVED**。各一次B32×20、32任务/640接受、退出0；相同8×4划分，已存本机进程wall 0.709351/8.041770s；`exact_history_contract_supported=false` 明确内部语义差异。没有把该时间无条件除成GPU硬件倍率。 |
| 真正支持入口：显式CPU/CUDA、限两任务、锁280且clean、数值失败非成功、证据保存 | 当前 `repo/experiments/review_suite/cli.py/test_cli.py`（本地implementation副本）；`runs/supported_entry_validation/{tests.log,cpu_checks.json,cpu_smoke.log,cuda_smoke.log,old_cpu_smoke.log}` | **PROVED**。4项入口回归OK，含原smoke失败删除证据问题的修复回归；两系统CPU/CUDA真实2步smoke均成功且source干净280；旧CPU路径2步兼容检查成功。CLI拒绝缺device、旧/dirty引擎和非两个固定任务。 |
| README真实可复现小运行；支持run/observe实际执行 | `docs/REPRODUCING.md`、adapter README；`runs/supported_entry_validation/vdp_full_cpu{.receipt.json,/summary.json}`、`vdp_full_observed{.receipt.json,/summary.json}` | **PROVED**。实现445的统一run实际VDP1000步，统一observe实际1000模型/4000行，两receipt退出0；另已验证CPU/CUDA两系统smoke。实际科学运行固定于445，最终文档HEAD由外部交付清单单独标明。 |
| 中文采用结论、全部最终耗时、维护/接入代价和总目标状态 | 已实际阅读全文本地 `REPORT.md`；`final_timing/runs/timing_analysis.json` 的全部pairs/aggregates；已交付入口/来源绑定/质量结果 | **PROVED（既定双系统交付）**。报告给出“值得继续复用”，采用280并保留CPU/CUDA显式选择、旧参考和默认行为；说明CPU控制/support/同步及SR回滚拷贝的剩余成本。六行配对的全部24个单次时间、四个逐对倍率中位数及八个CV已按报告精度逐项对照通过。报告区分迁移收益、纯设备对照与离线观察成本，披露共享主机、失败版本、作废冷启动尝试和有限数值证据；现有B32 VDP优先CPU，Bruss持续求解可用GPU。更大批量、第三系统/控制器及继续提高GPU收益作为未执行后续扩展，不用这些扩展支撑当前结论。 |

最终复核没有发现需要新增求解、GPU实验或大范围数值审计的缺口。唯一正式campaign的12样本已完成、验收通过；最终报告已完整纳入全部样本、计时边界/波动及路线判断。本审计科学/实现条目均有已读证据支持。根代理随后完成文档提交，并在外部 `R/DELIVERY_STATE.json` 写入最终文档HEAD及干净状态；该清单区分最终文档身份和固定测量实现445，避免本文自引用提交哈希。

数值质量仍有明确限制：同观察器不会消除内部表示和历史策略差异；最窄范围不能据此认定最正确；Flow*尾步不具匹配域；六例GPU反例、102项CPU回归和生命周期测试只建立各自有限证据。此次完成验收没有把这些结果升级为完整形式证明。
