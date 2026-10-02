# TORA remain：P3 的 `h=0.05` 隔离数值诊断

本轮只把原 P3 全初盒配置的 `ode_step_size` 从 `0.1` 改为 `0.05`；本地两份实际 `config.yaml` 的逐行差异只有这一行。官方 ONNX、12 个初盒、四态 ODE、每秒一次控制、20 秒时域、`[-2,2]^4` 全时安全带、Taylor order 3、余项估计和 CROWN 设置均沿用[固定 `h=0.1` 合同](../../../../ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md)。事前[意图](INTENT.json)规定新 run ID 与独立 1 秒门检；旧 `p3_tora_remain_v1/full20_001` 未重启或修改。

| 新作业 | 原始外层结果 | 数值覆盖 | 全初盒接受与保存安全带 | 外层 wall |
| --- | --- | --- | --- | ---: |
| [一期门检](smoke1_001/RESULT.json) | `completed/exit0` | 20/20 小步，`T=1` | 240/240 盒步接受，保存 tube 均在带内 | 5.465091 s |
| [独立全程](full20_001/RESULT.json) | `completed/exit0` | 400/400 小步，`T=20` | 4800/4800 盒步接受，保存 tube 均在带内 | 13.806753 s |

[独立扫描](full20_001/INDEPENDENT_INTERVAL_SCAN.json)读取 4,800 条原始 `ranges.bin`，确认小步与盒身份、`h=0.05`、所有接受区间有限有序、endpoint 在同段 tube 内、无保存 tube 越出安全带。`T=20` 的四态 endpoint union 分别为 `[-0.098188721,-0.026556506]`、`[-0.257092032,-0.133192073]`、`[0.371147899,0.704159726]`、`[-0.355924884,-0.001610474]`；全时 tube union 与观察器收据见[原始 payload](full20_001/payload/RESULT.json)。一期的[独立扫描](smoke1_001/INDEPENDENT_INTERVAL_SCAN.json)也无拒绝、越带或 endpoint 超同段 tube。

两次运行在物理 GPU 3、CPU 14–17 的新目录中，分别由 120 秒上限的无哈希 supervisor 管理。全程作者 checker 无文字输出；这里只有完整数值时域与已保存区间安全观察，没有独立端到端浮点 NNCS 证明。这个 `h=0.05` 结果是另列参数 profile，不填补固定 `h=0.1` 主表或提供四方同设置速度排名。

[实际执行入口副本](EXECUTED_ENTRY.py)中的首次盒拒绝会被观察记录，但原驱动仅在全部盒失败时自行停算；事前意图所写的“首拒即停”在此入口尚未强制执行。两次新运行均没有任何盒拒绝，因此这项差异未改变其 20/400 步结果。当前[维护入口](../../../../../tools/archcomp26_tora_remain_p3_nohash.py)已在记录写盘后增设首拒抛错，今后隔离失败尝试不会继续跑剩余小步；执行副本保持原样以供复核。
