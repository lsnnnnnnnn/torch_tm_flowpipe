# ARCH-COMP26 Attitude Control：修正 unsafe 集的原生 Flow* 新运行

合同：[2026 Attitude Control 新执行合同](../../../../ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。新建原生构建根 `N/runs/archcomp26_20261001/native_attitude_avoid_build_001`，完整运行根 `N/runs/archcomp26_20261001/native_attitude_avoid_full30_001`，其中 `N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。使用固定官方 2026 torch ONNX、全初始盒，唯一性质修正是把旧 `-x4 - 0.4` 改为官方 `-x4 - 0.7`；没有重启或改写旧实验，也没有计算内容摘要。

| 运行 | 覆盖 | 外层 wall | 进程输出 | 证据 |
| --- | --- | ---: | --- | --- |
| 一周期 smoke | 1/1 控制期、2/2 ODE 小段、1 RPC | 3.874486652 s | `PREFIX_SAFE_NO_FULL_PROPERTY`，只代表 `T=.1` 前缀 | [smoke RESULT](../native_attitude_avoid_smoke1_001/RESULT.json) |
| 全程 | 30/30 控制期、60/60 小段、30 RPC，单个完整初盒 | 6.281498344 s | `VERIFIED`；原生内部报告 2.786 s | [RESULT](RESULT.json)、[native.log](native.log) |

[构建记录](../native_attitude_avoid_build_001/BUILD.json)记录实际源、编译命令和字节大小；生成入口为仓库的 `tools/build_archcomp26_attitude_avoid_native_nohash.py`，运行入口为 `tools/run_archcomp26_attitude_avoid_native_pair.sh`。原生二进制留在服务器新目录。由该二进制导出的 [initial_boxes.json](initial_boxes.json) 数值上与官方六维全初盒加四个零辅助态完全相同。

[SCAN.json](SCAN.json) 对全部 60 条保存的六维 whole-segment tube 与 endpoint 记录检查：格点完整、无非有限数、无倒置区间、同段 endpoint 没有超出 tube。[BOX_UNSAFE_AUDIT.json](BOX_UNSAFE_AUDIT.json) 对每个保存 tube 的六维轴向盒直接检查，60/60 与修正后的官方 unsafe box 不相交。尤其 `x4` 全时域 tube union 为 `[-0.8235137167865756,-0.710800050132425]`，距 unsafe 下界 `-0.7` 仍有 `0.010800050132425` 的保存区间间隔。

| 坐标 | whole-time tube union | `T=3` endpoint | endpoint 宽度 |
| --- | --- | --- | ---: |
| `x1` | `[-0.4500015626338102,0.16976824354640618]` | `[0.16557407729671958,0.16976821175934212]` | 0.004194134463 |
| `x2` | `[-0.5500045505366273,-0.2704292183914348]` | `[-0.2764093720892744,-0.2704292183914348]` | 0.005980153698 |
| `x3` | `[-0.006193169414452644,0.6600063263115077]` | `[-0.006192916476587179,0.000001060436144673]` | 0.006193976913 |
| `x4` | `[-0.8235137167865756,-0.710800050132425]` | `[-0.7430921814946506,-0.710800050132425]` | 0.032292131362 |
| `x5` | `[0.5053188062469477,0.8677794035743642]` | `[0.5053635177534117,0.5229994605864844]` | 0.017635942833 |
| `x6` | `[-0.650024396604616,-0.04233931487778324]` | `[-0.06099215808457448,-0.04235475413672044]` | 0.018637403948 |

本页的直接盒分离检查是对**保存的原生区间数据**再判定，不是独立证明上游 CROWN 浮点包络或整个 NNCS 的数值可靠性。此单次进程时间也不能当稳定中位数。
