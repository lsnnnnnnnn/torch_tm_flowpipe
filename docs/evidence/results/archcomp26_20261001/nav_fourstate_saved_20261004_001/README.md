# NAV 两合同四方法：四态保存界与绝对宽度

这里只读已有 `ranges.bin`；不运行控制器、求解器或新 benchmark，不计算内容摘要。
`all_states_saved_curves.csv` 有 19200 行：standard/robust × 4 方法 × 600 步 × x/y/speed/heading。
`absolute_widths.csv` 有 32 行，逐态区分 T=6 终点、最后一步 tube、[0,6] 全 tube union，
并给终点每盒 mean/max 以及每个初盒整段时间包络的宽度 mean/max。
逐步表同时保留每盒 tube/endpoint mean/max；这些统计均来自所有真实保存分盒。

P3 是当前 working P3；Huan/Xiangru 是 2026-09-23 同合同历史；standard native 是 2026-10-02，robust native 是历史。
这不是四方法同时新跑的计时比较；完整保存数值时域不升级为独立浮点 NNCS 证书。
x/y 投影图仍使用原 `nav_current_p3_saved_20261004_001/`，新增本表补齐其它物理态。
所有 8 源逐条检查有限、顺序、600 步完整、h=.01（152字节记录）、endpoint 包含在本段 tube。
源路径、字节数、真实记录数与布局见 `AUDIT.json`；精简原始派生 JSON 在其 `source_json`。
时间单位为秒，状态单位在来源合同中未显式声明，未换算。
