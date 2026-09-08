本轮在最终端点修复上研究一次 attempt 内的固定多项式工作复用。开关默认关闭：

```python
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
with prepared_remainder_replay(True):
    segment = existing_frozen_step(...)
```

`run.py` 复用上轮冻结合同、scheduler、common observer 和端点导出方式；新增
完整状态摘要、精确时间、独立 solve 计时事件和 prepared 模式身份。准备成本仍
在每步 solve 计时内。`replay.py` 从真实步骤取得同一个候选和已接受余项，独立
运行两条完整收紧循环，再逐轮比较输入 R、proposal、所有账本、系数和决策。
它的证据开销不进入性能分母。`profile.py` 单独输出互斥函数时间以及另一个
步骤的 cProfile self time；inclusive time 不能相加。

干净提交、原 py11、CPU 单线程与同一 affinity 下执行：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.repaired_solver_performance.launch --cpu 2 --output /absolute/new/run/directory
```

协调器只执行一张有限的顺序任务表，保存每个子进程 PID、启动 ticks、真实命令、
退出码和整个进程耗时。观察连接中断不是任务失败。再次启动只跳过已证实成功
的任务，遇到未完成记录主动停止，不能凭锁文件或日志缺失重复启动长跑。

窗口均来自初始完整状态或合法 checkpoint。缺少 VDP90，故从上轮20推进一次
70步；缺少 Brusselator980，故在本轮一次完整 reference1000 中保存后补做
981–1000 窗口。这是缺失检查点造成的测量顺序调整，不额外进行一遍长轨迹。
各窗口和100步前缀至少3对、交替先后顺序；固定完整各一对，自适应新优化只做
合同/状态一致性，不主张自适应速度。

正式长跑固定在 `e3e69138c1cf91018f301d180c5f9f5223720131`。若自行重跑，
先在该提交创建干净 checkout，再执行上述协调器；最终 package 的提交另有
分析、验证器与报告，不能把它标成已有长跑的来源。

交付包的有界复核命令如下，不重跑完整 ODE 长轨迹：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.repaired_solver_performance.verify \
  artifacts/runs/repaired_solver_performance_20260908T034636Z
```

验证器重新执行保存的局部独立 replay 循环，核对完整范围、h、时域、队列/端点
摘要和实际计时事件，再推导 CSV、倍率与最终状态。源码/来源检查单列，不能
替代数学证明。`tests/test_evidence.py` 的六种语义篡改在更新外层 hash 后仍须
拒绝。`build.py` 只接受完成的正式任务表，生成数值汇总与 PNG/PDF 图。

`run_tests.py --phase targeted|matrix|evidence --output /absolute/test/logs`
保存原始命令、退出码与 JUnit。矩阵沿用父版本九组测试的精确路由：旧停止
测试在 `73c3b48`，旧端点证据在 `7e41f33`，其余在当前 package；恢复的历史
fixture 逐文件核对原 25 个摘要。`evidence` 阶段还需 `--artifact` 指定已构建的
完整包。完整矩阵、历史证据测试和六个新证据测试按测试身份去重，targeted 与
独立 clone 的重复检查不加总。测试日志、源码与报告的实际验收记录才决定最终结论。
