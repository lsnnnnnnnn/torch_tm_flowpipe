# 目标逐条验收

| 要求 | 证据与状态 |
|---|---|
| 冻结与版本 | PASS：运行科学 SHA 1d870939fcdb72bd999730cf09de0b14d1cd71c4；计划先于正式执行冻结；离线验证器修正 SHA 0b0da13da2c563ddf287fa79731e2c8653d46852 有独立衔接收据 |
| 真实异构 packet | PASS：连续描述符、每请求私有链、一包四 kernel、2 H2D/2 D2H |
| 所有权/取消/epoch/cap | PASS：设备所有权检查、私有 scatter、固定上限、安全切包与显式回退 |
| 有限正确性矩阵 | PASS：B1/B2/B8/B32、B2×120、历史清空及独立 Fraction 捕获请求检查 |
| 父碎片机会 | PASS：只读 REUSED 生命周期，区分 KEY_SPLIT/ARRIVAL_LIMITED，不作可加速声称 |
| 五区块正式性能 | PASS：40 次新鲜 run；效果 no_gain；目标 False |
| 原始 B1 长时域 | PASS：VDP T10 与 Brusselator T20 各 1000 步，无保存答案/CPU 周期纠偏；Brussels owner-schema 验证器修正未重跑计算 |
| 共同 observer | PASS：当前严格 observer 重跑 REUSED CPU/Flow* 完整对象，近零及>1.10分列 |
| 主张边界 | PASS：默认关闭；非全 GPU；非全求解器形式化证明；不声称全仓 pow_int 修复 |
| 篡改与独立 clone | 由 tamper/ 与 acceptance/ 实际回执关闭；独立副本只做有界 B2×2 重跑 |
| 推送 | 仅推送本次独立分支，不改 main、不强推；最终 local/remote/clone SHA 另存实际回执 |

所有 PASS 均由原始文件和语义验证器重算；外层 hash 不是唯一检查。
