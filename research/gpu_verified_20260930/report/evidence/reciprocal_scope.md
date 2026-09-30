# Huan / Xiangru 归档版本倒数反例：实际 CPU 复现

两份未修改的作者归档源码均实际复现了倒数外包失效：对 `1/(2+z/4)`、`z∈[-1,1]`，在 `z=-1` 的精确值是 `4/7`，但 1 至 4 阶完整返回区间的上界都更小。所有运行的 `bad` 标志均为 false。

每位作者使用一个全新本地 CPU 进程，直接导入各自归档树。每份树的全部 29 个 Python 文件均匹配原交付 manifest 和已保存 ARCH CartPole 真实运行记录的 engine SHA；全部已导入的 flowstar_gpu 模块也逐项验证路径与 SHA。没有修改作者文件、替换我方模块、编译或初始化 CUDA。

| 版本 | 实际执行的 commit | CPU 过程时间 |
|---|---|---:|
| huan | `d5f0b68fcd36ba5f582733624f074728fe9720d8` | 1.245237 s |
| xiangru | `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3` | 1.244810 s |

两位作者这 4 组输出的全部保存字段逐字节一致。下表对两者均适用；差额使用实际返回的 binary64 系数区间和完整 RHS ordinary remainder，通过 Fraction 精确重算。

| 阶数 k | 返回外包的上界 | 精确值 4/7 超出上界 |
|---:|---:|---:|
| 1 | 0.56833090379008833 | 0.00309766763848 |
| 2 | 0.5711454862557277 | 0.000283085172844 |
| 3 | 0.57140806053653326 | 2.05108920382e-05 |
| 4 | 0.57142813253200553 | 4.3889656593e-07 |

配置：n=1、B=1、binary64、CPU、cutoff=0、零输入余项；表基阶数为 max(k,2)，执行器级数阶数为 k。保存了实际 field、ordinary remainder、cache、strict_tails、支持和指数。CPU 审计另行加载全部 8 个 .pt，重算并核对上述 8 个有理数反例。

`strict_tails` 是 VAR 被截断项的旁路缓存，已经通过输入余项进入最终 ordinary remainder，不应另加一次；该仿射初值没有被截断的 VAR 项，实际 strict_tails 全为零。本检查没有遗漏或重复累加它。

源码位置：两者 elementary.py:1152–1177 均先建立归一化有限级数，1175 行调用旧 Taylor 尾项，1176 行再乘 rec_c。实际 sparse 调用分别位于 Huan sparse_exec.py:634–646、Xiangru sparse_exec.py:656–668。VAR 尾项进入 ordinary 并另存缓存，分别见 Huan:597–598、Xiangru:619–620。

范围：这是 Huan `d5f0b68`、Xiangru `1c16d4e` 的 CPU sparse 通用倒数路径反例，不是全部 branch 的结论，也不是新的 CUDA 复现。这里中心 c=2；不能由此直接断定当前 QUAD 的 cosine 分母、某次实验外包或最终安全结论已经失败。已有实验之间数值一致，也不能替代此数学正确性检查。

完整来源 SHA、实际输出与逐项复核见各作者 RESULT.json / order1..4.pt 以及 AUDIT.json。
