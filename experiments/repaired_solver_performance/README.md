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
PYTHONPATH=src:. python -m experiments.repaired_solver_performance.replay_evidence \
  --plant brusselator --steps 20 --output /path/to/replay_early.jsonl.gz
PYTHONPATH=src:. python -m pytest -q tests/test_prepared_remainder_replay.py
```

旧端点包的 verifier 包含其固定源码身份，必须在父版本 snapshot 执行，不修改旧 verifier 适配新代码。新结果的最终状态要由完整范围比较和本轮匹配生产时间推导；局部加速及测试数量不决定最终状态。
