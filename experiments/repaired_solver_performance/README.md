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

整个任务还需从原始记录生成新比较器、汇总、图与中文报告，完成全矩阵和独立
clone 复核后才可下最终状态结论。此说明不以局部测试代替长跑验收。
