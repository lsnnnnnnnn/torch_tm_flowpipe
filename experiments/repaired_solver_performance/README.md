本目录在最终端点修复 reference 上测量同一步准备复用。数学配置直接使用已提交的 `experiments.endpoint_roundoff_repair.frozen` 和固定摘要的 `MATCHED_CONTRACTS.json`。

正式运行必须在干净源码 worktree 中，使用原 py11、CPU float64、单线程和相同 affinity。以下 `SHA` 必须是运行目录的实际 HEAD，输出目录必须不存在：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 3 /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.repaired_solver_performance.run \
  --plant brusselator --steps 1000 --mode reference \
  --scientific-sha SHA --output /path/to/fresh_reference
```

`--mode prepared_remainder_replay` 启用候选；`--plant van_der_pol --adaptive` 使用原自适应 scheduler。窗口通过 `--checkpoint /path/to/checkpoint_0100 --steps 20` 恢复完整状态，不从发布的盒子初始化。准备和必要保证检查始终在 solve 中；原完整对象、上下界、E 和状态/队列导出在 solve 外。调用进程的外部计时另外记录 Python 启动、初始化和全部 I/O。

`replay_evidence` 保存真实候选及每轮 R，同时比较同输入 evaluator 与独立完整 replay loop。这是带证据导出的正确性实验，时间不作为生产性能分母：

```bash
PYTHONPATH=src:. /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.repaired_solver_performance.replay_evidence \
  --plant brusselator --steps 20 --output /path/to/replay_early.jsonl.gz
PYTHONPATH=src:. /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m pytest -q tests/test_prepared_remainder_replay.py
```

旧端点包的 verifier 包含其固定源码身份，必须在父版本 snapshot 执行，不修改旧 verifier 适配新代码。新结果的最终状态要由完整范围比较和本轮匹配生产时间推导；局部加速及测试数量不决定最终状态。

`benchmark` 是有限的正式序列：五个 20 步窗口各三对，两个 100 步前缀各三对，然后两套完整优化运行和一次 VDP 自适应运行。它要求两条 fresh reference 已结束，并从其完整 checkpoint 恢复窗口；每对交替执行顺序。所有 production 子进程串行固定 CPU 3，外部 whole-process 时间、负载与退出码写入 `commands.json`。活动任务通过 `active_process.json` 和实际进程状态核对，已有输出不会自动覆盖或重启。

本轮完整 reference 科学提交是 `e2d00f1`，与最终修复 `0714e47` 的数值 src 完全相同。正式窗口、前缀及优化长跑使用 `f627d64`；它的数值 src 与完整矩阵测试过的 `ae6e21a` 完全相同。之后的比较器、绘图和 package 提交单独归属，不作为这些长跑的源码身份。

初始完整优化运行开始时，宿主负载已从最初 reference 的约 1–2 升至约 40，未修改的 VDP/reference 路径也出现明显时间波动。因此在第一条完整优化结果产生之前，另行登记两套系统各一对相邻的完整复测：Brusselator reference→优化，VDP 优化→reference。`confirm --work-root ROOT --source CANDIDATE_SOURCE` 在原 45 条子运行结束后顺序执行这四条运行，仍使用干净 `f627d64`、CPU 3 和相同数学合同。这些相邻配对作为主要完整时间比较；最初的全部完整结果、负载和数值桥接仍保留并单独作图。复测数量预先固定，不按有利结果追加。

已交付的包可以从任意包含本分支历史的 clone 进行有限复核，不会重新运行完整 ODE 长实验：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 6 /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.repaired_solver_performance.verify \
  artifacts/runs/repaired_solver_performance_20260908T084224Z \
  --output /tmp/repaired-performance-verification.json
PYTHONPATH=src:. /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m pytest -q tests/test_prepared_remainder_replay.py \
  tests/test_our_reference_endpoint_containment.py \
  tests/test_endpoint_roundoff_repair.py tests/test_endpoint_roundoff_carry.py \
  experiments/repaired_solver_performance/tests
```

完整 verifier 先核对身份、配置、原始计时和逐步等价，再从 100 个真实步骤的输入重算 668 次 proposal 与两条独立 loop，并重新计算完整 reference 和自适应保存模型的端点 E 与上下界；优化固定运行的全模型与 reference 逐位相同。旧 Fraction/传递回归直接复用。`--structural-only` 仅供六种重新散列后的语义篡改测试使用，不能代替最终完整复核。

`package --work-root ROOT --output ARTIFACT` 只接受已完成的 production、确认复测和 profile。`profile_windows.csv` 只汇总互斥的子 scope 扣除时间；`remaining_hotspots.csv` 使用另一轮 cProfile self time。初始化、数值 solve、导出和 whole process 单列；准备秒数是 solve 的内部子项。窗口恢复的 checkpoint 读取也单列。复用 Flow* 的共同 observer 宽度和历史时间均不构成本轮 fresh 性能分母。
