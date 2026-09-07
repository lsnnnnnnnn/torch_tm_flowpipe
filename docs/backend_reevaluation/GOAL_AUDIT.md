# 逐项完成审查

本轮完整执行目标的停止分支：在复用前提核对中确认第三种独立端点漏包，依据第 4、8 节停止候选数值开发。未开启的阶段不是通过，既有两步失败诊断也没有被重写。

| 目标条款 | 当前权威证据及结论 |
|---|---|
| 0、1 固定起点与冻结旧证据 | 新 worktree 从 `3ca31b8` 建立，ROOT/BRANCH 与完整版本写入 RUN_CONTEXT/SOURCE_MAP。旧 verifier 已从已提交原始对象复算两例和七表，退出码 0。末次只读检查见 `raw_minimal/source_end_check.json`；旧证据、旧文档与 `src` 相对 base 无 diff。未再次 clone 第三方源码，未查询变化中的上游 tip。 |
| 2 禁止事项 | 没有数值源码补丁、默认入口替换、第三方推送、自动合并、额外系统长跑或新 GPU 架构。没有通过改 CPU、容差、盒子或合同帮助候选通过。现有项目回归组照原完整测试合同执行，不是新增 benchmark。 |
| 3 三份实现同例复查 | WITNESS_CROSSCHECK 共 51 行；原始历史矩阵直接来自旧 JSON，另外检查非单位尺度构造。归一化采用零常数两状态仿射例，并按整个像集计算。A/B 实际 CPU/CUDA dense/sparse 与普通/严格分派均记录，C 用实际 CPU 操作。A 失败，B strict 与 C 通过；不能推广成全局正确性结论。 |
| 4 修复传承与局部性 | REPAIR_REUSE_PLAN 回答五个问题；已有区间 Phi/逆尺度/误差余项保护在 B 的真实入口存在，A 缺失。独立端点输入的 8 行 CPU/CUDA dense/sparse 结果触发停止，记录在 endpoint_A/B.json；B 源码读取到此停止。 |
| 5 新候选与许可 | 条件未满足，未建立修复候选 worktree，补丁 SHA/包均为 null。只记录第三方路径、版本、摘要和 GPL-3.0-only 来源说明；本分支没有第三方源码、diff、二进制或环境。静态 fresh-clone 验证不意味着获得私有第三方源码权限。 |
| 6、7 两条完整误差链 | 局部旧机制通过，但完整当前 J 所有权等尚未闭合。未把旧局部通过重标为修复候选通过，未实施当前 J、极小数或分派修补。NUMERICAL_CHANGE_CONTRACT 明列误差应如何计算、保存和消费。 |
| 8 安全门与停止 | 原始两例、增加尺度后的新矩阵、旧 1/2/3 维历史及清空测试已执行。第三端点反例不经过历史/缩放，精确值为 `3/144115188075855872`，A 返回 `[0,0]`。其余新候选安全回归因停止条件未开启，未宣称完成整条严格路径。 |
| 9 参数合同 | MATCHED_CONTRACTS 与上轮逐字节相同，未改两系统方程、初始集、步长/阶数/队列/预算，未增加时钟。没有接入新后端，未假装 validation_eps 的接口语义相同，未开发 adaptive+SR。 |
| 10 成本递增顺序 | 修复候选的 1–2/20/120 步与完整 T20/T10 均没有启动；LONG_PREFIX_STATE_CHECKS 逐阶段说明原因。没有重复完整 CPU 长跑。冻结原始数值结果继续保留为 reused；未进入第 10 节的新运行环境/性能比较阶段。 |
| 11 宽度硬门 | 三张宽度表四项分别记为 NOT_STARTED；没有新全程宽度图，figures/README 解释缺失。原 1.10 候选门与 Flow* P95/最大比率门均未放宽、未授予 PASS。 |
| 12 批量性能 | 新 B1/B8/B32 无性能数据，timings_raw 明确未测；四项旧 B1 单次时间从原始 summary 重建并标 REUSED_FROZEN。没有 OOM 降级、外推或 microkernel 代替完整求解。速度门保持未建立。 |
| 13、14 结果与主线 | RESULT 的停止状态由精确 raw data 导出，ADOPTION_DECISION 保留 CPU 默认能力并说明回到既有 profile 和重复调度分析；不等待监控、不发送维护者材料、不启动新 CUDA 或其他后端工程。 |
| 15 紧凑交付与复核 | 五份主文档、所有指定 CSV/JSON、raw_minimal/tests/figures 与 SHA256SUMS 齐全；不适用数据有明确原因。旧 verifier 的 hash/表格工具被复用，新 verifier 重算包含性、七表和停止决定。直接篡改输入、输出、模式、SHA、计时及决定均被拒绝。日志、JUnit、退出码和命令保留；旧资料缺失不作恢复。 |
| 16 白话九问 | REPORT_PLAIN_CHINESE 中逐项回答：局部失败与长轨迹区分、修复资产、CPU 对称结果、实际改动、未开启时域/宽度/速度、推荐为空与下一步方向。函数名和 SHA 放附录。 |
| 17 提交纪律 | 科学驱动 SHA、上游 SHA、旧修复 SHA 与封装版本分开；本地候选补丁为 NONE。只发布用户自己的新 torch_tm_flowpipe 分支。最终 tip、独立 clone 和已提交证据核对记录见本轮 ROOT 下 `final_delivery.json`，不会把自引用提交 SHA 塞回数值证据。 |

## 测试与发布记录

完整根目录测试来自固定代码提交 `96de71f7e02e1d8280f208c716179d93efe79ab5` 的完整 SHA（以 tests/commands.json 为准）：**954 通过、2 跳过，450.12 秒，退出码 0**。新增的 9 项证据测试包含在这 954 项中；其中 8 项直接篡改在重算外层哈希后仍被拒绝。旧候选相关历史测试另有 4 项通过。

项目完整脚本要求的七组隔离回归共 **59 通过、9 跳过、1 失败**。唯一失败为 `test_frozen_historical_result_manifest_is_unchanged`，原因是新工作树没有收录 `experiments/three_way_common_contract/results/20260724T132534Z` 的旧结果目录；未修改相关源码或断言，依目标第 15 节记录后停止恢复旧资料。项目不同用例合计 1013 通过、11 跳过、1 项历史资料检查失败，不能称为全绿。全部 JUnit、原始日志、实际退出码及每组命令见 `tests/TEST_EXECUTIONS.json` 与相邻文件。

已从用户 GitHub 新分支建立独立对象库的 fresh clone，核验封装提交 `78a3e2757b2aa7252c428fe23f143328a6b307b1` 成功：51 行三方精确映射、8 行独立端点观察、七张汇总表全部重算通过；记录已收入 tests/independent_initial.json、日志与退出码。本次最终封装后的再次核对保存在本轮 ROOT 下 final_delivery.json。它记录远端 tip、本地 HEAD、独立 clone HEAD 和 verifier 输出，封装提交不充当数值算法 SHA。
