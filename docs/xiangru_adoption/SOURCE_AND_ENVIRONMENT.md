# 源码与环境

新克隆来自 GitHub 正常仓库 URL 和用户现有授权。2026-09-07 03:25:52–03:25:55 UTC fetch，`2026_experiment` 当时 tip 与运行 SHA 均为 `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3`。目录为 `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream`，实际 checkout 为 detached。旧 Xiangru 为 `84184de6c2b3f1ff2da6755f732d91925037025d`；新版增加仓库内 `src/flowstar_gpu`、来源许可文件和测试。旧 Xiangru 工作树干净，没有需要搬运的未提交补丁。

仓库内引擎来源标注 Huan `flowstar-gpu` 的 `7a8002df4d45297c915c67f75247a0b88a8e0b03`，声明许可保留为 GPL-3.0-only，作者 Huan Zhang。导入 manifest 是原始导入的记录，不是后续所有文件必须相同的要求。没有把第三方源码或修改 diff 推送到用户仓库。

对照证据起点为 `df50c55ef640b0ca9d90b132c88235b8ec07510b`；我们的实际数值运行来自独立 worktree `our_optimized` 的 `4939fb288c941a67f55cc191f4d75f8594692f47`，CPU 任务包装另固定在 `7608dd52e48af3ce8ae2e0a8343aae125c63b7f4`。用户远端 main 当时另有 `b2f34f5b2077e34662a2559d8c09b1d264bd7d98`，没有替换本轮指定参考。审计分支的 numerical `src/` 相对 df50c55 无改动。

原生 Flow* 数值基准为 `b85a3211748cb77b736fe4ad42ee02d8d2b81148`。新的独立目录加两个 const 只读 term 访问器，本地观察提交 `722a5611c0b33564b11916b13474d0d21d74bc0c`；重新构建，未更改数值语句。两状态原生驱动与 Brusselator CPU 运行器来自独立且干净的 scientific worktree `runner_scientific`，固定在 `c1326f7549f20ae9036291386a4a65e575f5f488`。VDP 启动前修正运行器末步按浮点剩余时间截短的问题，保持每步严格为 0.01；VDP 使用另一个干净 worktree `runner_vdp_scientific` 的 `901029fdd395dc69c9357ad911b594250b4240cc`，未改求解器数值 src，也未修改正在运行的 Brusselator worktree。正式数值版本与后来的报告封装提交分开记录。

机器为 `huan-c4140-server-3`，Xeon Gold 6138，80 个逻辑 CPU。本轮 GPU 为设备 0 的 Tesla V100-SXM2-16GB，sm_70，驱动 580.159.03。CPU 计时单线程、亲和性 CPU 2；短正确性检查和现有仓库测试在其他核执行（主要 CPU 4/5），所以本轮单次 CPU 结果不声称独占整台机器或具有重复运行方差估计。没有终止其他用户任务，也没有读取完整进程参数。

我们的环境保留原 `py11`：Python 3.11.15、PyTorch 2.5.1+cu121。候选在全新 venv：Python 3.11.15、PyTorch 2.5.1+cu124。只安装 torch、numpy、pytest、hypothesis、ninja、mpmath 等必要依赖，以及测试读取配置需要的 PyYAML；没有安装完整 NN/ONNX/CROWN 工具链。候选包实际导入路径严格来自本次 `xiangru_upstream/src/flowstar_gpu`；环境、编译和逐进程来源记录见 source_manifest 与 raw summaries。

CUDA toolkit 为 12.6；系统 GCC 15 不适合该 nvcc 的头文件解析，使用已有独立 GCC 13 工具链，并通过上游已经支持的 `FLOWSTAR_CUDA_HOST_COMPILER` 配置。架构由上游读取实际 V100 capability 得到 sm_70。extension cache、Inductor、Triton、Hypothesis 存储均位于本轮目录；没有复用旧来源不明的 `.so`。首次 segment extension 构建 43.218453059 秒；tape extension 单独在短诊断中编译，不混入完整求解性能。完整构建参数和实际 `.so` 路径另存。

生产入口的默认引擎为 sparse；Settings 默认 strict，但集成 CLI 默认通过 `--strict` 与否选择 parity。所有本轮调用都显式指定 mode。稀疏预处理的 GLUE_MODE 默认 eager，且 B<64 总会执行同一 eager helper，因此 B1/B8/B32 的缩放回放并非旁边的另一个修复实现。VDP 自带 golden 是五阶且多一个时钟状态，Brusselator golden 也有时钟状态，本轮没有直接替代使用它们。

当前 Settings 仍明确拒绝 adaptive + symbolic remainder。没有借此关闭历史队列或新写自适应功能。CPU eager、CUDA dense、CUDA sparse 分别有短诊断；历史传播另外通过 profiler 记录实际 CUDA 矩阵和区间 kernel，不能仅由 `cuda.is_available` 或 extension 加载推断执行路径。

原生 Flow* 的跟踪源码与观察提交一致；其构建目录保留未跟踪的对象文件、静态库和由解析器生成的文件。这些产物单独记录在 `raw_minimal/live_source_check.json`，没有伪装成源码修改或推送的第三方文件。候选与两组 scientific 运行器均保持工作树干净。
