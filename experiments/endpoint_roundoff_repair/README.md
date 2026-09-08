本目录只处理端点常数代入修复的复核。`verify.py` 重算保存证据中的精确系数包含、完整端点包含、实际时间、宽度、测试退出码和结果状态，同时回放小型真实传递路径。它不会重跑两套 1000 步 ODE。

在仓库根目录、已安装 torch/PyYAML/matplotlib/pytest 的环境中运行：

```bash
PYTHONPATH=src:. python -m experiments.endpoint_roundoff_repair.verify artifacts/runs/endpoint_roundoff_repair_20260908
PYTHONPATH=src:. python -m pytest -q tests/test_endpoint_roundoff_repair.py tests/test_endpoint_roundoff_carry.py tests/test_our_reference_endpoint_containment.py
PYTHONPATH=src:. python -m pytest -q experiments/endpoint_roundoff_repair/tests/test_evidence.py
```

数学长跑使用干净提交 `196a50e9131336d68df07ad0af353deca0092d19`。在该提交的独立干净 worktree、原 py11 环境、CPU 单线程下，以下命令才属于重新长跑；每个输出目录必须尚不存在：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.endpoint_roundoff_repair.run --plant van_der_pol --steps 1000 --scientific-sha 196a50e9131336d68df07ad0af353deca0092d19 --output /tmp/repaired_vdp_fresh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.endpoint_roundoff_repair.run --plant brusselator --steps 1000 --scientific-sha 196a50e9131336d68df07ad0af353deca0092d19 --output /tmp/repaired_brusselator_fresh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.endpoint_roundoff_repair.run --plant van_der_pol --adaptive --scientific-sha 196a50e9131336d68df07ad0af353deca0092d19 --output /tmp/repaired_vdp_adaptive_fresh
```

交付运行时 `0714e475ed9e73bec31619c9c690d1fd63de3d36` 补齐设备/精度兼容和稠密分量重组的类别传递；它与长跑提交的 CPU binary64 衔接证明、每个保存端点的逐位回放和实际检查点恢复结果位于 `raw_minimal/runtime_bridge.json`。封装提交继续保留这些不同身份。

根目录旧停止验证器只在旧 `73c3b48...` worktree 运行；其历史职责包括检查旧源文件和旧失败。全套矩阵的已恢复忽略夹具及 DiffReach 隔离环境按旧摘要复用，详见本次 `tests/*/full_commands.json` 和 `raw_minimal/historical_fixture_recovery.json`。独立克隆复核运行端点局部测试，不宣称在克隆中恢复了全部历史测试环境。
