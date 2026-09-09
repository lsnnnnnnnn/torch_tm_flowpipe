# 目标逐条验收

原目标 `goal_vdp_terminal.md`（摘要见SOURCE_MAP）保持全部范围。下列数值/测量项来自完整矩阵；独立副本和推送只有实际回执才能关闭。

| 要求 | 证据与状态 |
| --- | --- |
| §1–2 固定起点、旧树不改、独立分支 | 父b60a608；科学源码ef4e2f0c17518a4c4989071a596cd8631235c5ea；独立新分支，未reset/clean旧树、未改main |
| §2 指定阅读与父局部核验 | 指定文档/代码已读；父完整有界verifier与131局部测试在父worktree通过，标REUSED |
| §2 环境和导入 | 原py11 / torch2.5.1+cu121 / V100 GPU0 / CPU2；逐run保留实际导入路径、线程与环境；未升级工具链 |
| §3 数学参数与禁止范围 | 原方程/盒/阶数/cutoff/余项/E/收紧/历史/归一化未改；主VDP h=.01、Brusselator h=.02；CUDA .cu与440714逐字节相同 |
| §4 L/S/Q/G/S_gpu | 五条真实路线；正式S/Q/G同任务三组，L同工作量有限计时；S_gpu仅正确性；L保留已知幂局限 |
| §5.1 当前真实请求、worker上下文 | 真实step内阻塞Future；显式ContextVar；一个服务线程拥有CUDA stream/event；不读取保存答案 |
| §5.2 完整身份 | run/task/epoch/generation/attempt/counter/完整key；诊断输入及源状态hash；每任务最多一条未消费请求；生产身份检查保留并计时 |
| §5.3 语义分组与覆盖 | 复用有序支持/类型/角色/输出/dtype/device/外供表key；不补零重排；外供表显式CPU回退；coverage保留未覆盖调用栈，dense数学仍在CPU |
| §5.4 有界派发与策略冻结 | max_group、all_waiting、timeout三种派发；预注册2/20ms候选选20ms；正式不再调参；actual_grouping/flush_reasons/grouping_summary保留真实分布 |
| §6 请求错误与非有限 | 异常只影响相应任务；NaN/overflow不成功、不重用旧答案；有真实accepted step后的拒绝/重试证据 |
| §6 取消/迟到/恢复 | 取消与commit串行化；只保存最后接受状态；恢复新epoch，包括同runid跨进程；旧返回被丢弃，健康peer继续 |
| §6 完整carry与alias | 完整segment/下一状态比较；私有tensor输入输出；旧checkpoint丢私有诊断与顺序的接入缺陷已修复，保留开发反例；无线程/Future/CUDA指针pickle |
| §7 原算术与自检 | 四个原CUDA kernel不变；极小数/相消/三次幂启动自检；S/Q精确校正、G逐操作有向舍入；未改旧公开pow默认 |
| §8.1 固定子盒与原B1分列 | PARTITION_REFERENCE原8×4、固定B8/B1；B2任务0/31；原未分区B1独立标签和表；没有32份相同输入 |
| §8.2.1 B1/B2 | 两系统各2步，S/Q和S_gpu/G完整对象一致；actual raw+生命周期保留 |
| §8.2.2 B8 | 两系统S/Q/S_gpu/G均每任务20步；逐任务160完整segment同后端一致 |
| §8.2.3 B32 | 两系统S/Q/G每任务20步；S/Q全640段比较，S_gpu/G首两步64段比较；首两步实际请求完整精确核验 |
| §8.2.4 长依赖120 | 两系统任务0/31分别S/Q/G连续120步；每步来自自身上一步状态；VDP跨SR100清空；审计点1/2/60/100/119/120 |
| §8.2.5 历史窗口 | 安全完整checkpoint的VDP99/100/101与Brusselator999/1000/1001；四路线本地窗口，RESUMED_LOCAL_WINDOW；不是新GPU初始到1000步 |
| §8.2.6 原未分区B1 | 两系统S/G各20步并以共同observer比较真实范围；带诊断成本的延迟与子盒正式样本分列 |
| §8.3 异构专项 | 不同支持/项序、测试h、实际收紧2/5与请求137/215、accepted后异常、早完成/取消、timeout、小组、硬件CPU重算、恢复旧响应均有原始证据 |
| §9.1 完整行为与拆批 | 完整segment/endpoint/reset/E/映射/余项/owner/Phi/J/generation/收紧/停止计数/h/时间；B8拆2×B4、4×B2、8×B1及逆序/延迟/不同合法flush按task比较 |
| §9.2 独立算术审计 | 对各自实际输入核验幂/项/总和并重算后端；B32首2全量、故障成功边界、特殊结构/纠正/回退、长期预注册点；其他请求明确依靠旧局部合同 |
| §9.2 共同observer与警戒 | 实际endpoint/tube x/y上下界、精确累计时间、绝对/中心差、宽度比P50/P95/max及最差位置；≤1e-10分列；>1.10共0条，全部定位记录 |
| §9.3 保证边界 | 不宣称整个ODE形式化证明，不宣称新完整GPU1000步；历史长时域和旧自适应结果仍REUSED |
| §10.1 同资源 | 主S/Q/G CPU2单核、Torch intra/inter-op1、同GPU0；CPU3补充诊断不进入正式样本；正式序列逐子进程隔离 |
| §10.2 完整wall与冷启动 | 建初始状态到全部worker/服务清理计时；同步/必要校正/传输/检查均含；编译加载与自检另记；独立Fraction审计/序列化在外，必要CPU精确幂校正仍在内 |
| §10.2 合法时间线 | 真实服务/worker步骤区间并集的互斥分类与重叠；不相加并发等待当wall；主机kernel-and-sync不冒充设备纯kernel时间 |
| §10.3 重复和指标 | 60次正式实跑：B1三组2步，B8/B32三组20步，S-Q-G/G-Q-S/Q-S-G交替与6个L有限样本；所有成功分子、请求/分组/回退/等待/CPU时间/RSS/Torch显存已列 |
| §10.4 性能判定 | 922e3b3预注册两系统B32 min(S,Q)/G至少2/3胜且median≥1.10；实测状态LIVE_RANGE_SCHEDULER_VALIDATED__END_TO_END_PREFIX_SPEEDUP；G/L单列，Amdahl使用本轮已接入evaluator份额并明确排除未覆盖范围 |
| §11 修正与源码冻结 | 仅接入/调度/身份/缓冲所有权/保存恢复/证据层；正式数值源码ef4e2f0先提交，运行期间干净不变；包装代码另列SOURCE_MAP |
| §12 路线判断与机器字段 | REPORT第7项和RESULT明确最终路线；online/new full long horizon/formal solver proof/default enabled各自字段，速度不达标也完整交付 |
| §13 原始证据与重算 | raw_minimal/INDEX引用全部实际压缩模型/队列/请求/回执/时钟；独立重算派生CSV/决策到临时目录比较，不覆盖被验文件；SHA256SUMS涵盖所有新证据 |
| §13 篡改与测试身份 | 九类重新计算外层hash后的语义篡改被拒绝；根1221/2，最终28同身份不相加，父131 REUSED；三历史源码锁定模块留旧worktree，未弱化断言或加新skip |
| §13 独立副本有界真实重跑 | PENDING：必须独立git clone运行新package verifier及两系统S/Q/S_gpu/G B2两步实际重跑；不能以本地验证代替 |
| §13 最终分支推送与SHA | PENDING：按原授权推送自己的新分支，不强推/不改main；最后核对local/remote/独立副本SHA并保存实际回执 |

测试身份和每项原始路径见测试provenance、实验README及完整报告。包装过程不重算或覆盖科学性能样本。
