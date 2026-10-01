# 2026 TORA reach 控制器数值预检（无哈希）

日期：2026-10-01。范围仅为控制器文件、逐层参数和有限个点的前向计算；没有启动 reachability 实验，也没有决定新版主表的闭环合同。来源冲突见 [TORA reach 来源审计](ARCHCOMP26_TORA_REACH_SOURCE_CONFLICT_20261001.md)。

## 已核实的控制器关系

固定官方 2026 `nn_tora_sigmoid.mat` 与服务器旧 2024 `nn_tora_sigmoid.onnx`，固定官方 2026 `nn_tora_relu_tanh.mat` 与对应旧 ONNX，均是四层 `4→20→20→20→1` 网络。两组各八个权重/偏置张量在统一 MATLAB 末层向量形状后逐元素相等，所有形状一致。官方 `.mat` 激活序列与旧 ONNX 算子序列也相同：

| 控制器 | `.mat` / 旧 ONNX 激活 | 22 点独立 NumPy 与 ONNX Runtime 最大绝对输出差 | 初始盒中心原始网络值 |
| --- | --- | ---: | ---: |
| sigmoid | `sigmoid,sigmoid,sigmoid,sigmoid` | `1.1102230246251565e-16` | `0.461650623936015` |
| ReLU/tanh | `relu,relu,relu,tanh` | `9.020562075079397e-17` | `-0.047366557782876205` |

22 点包括完整初始盒 `[-0.77,-0.75]×[-0.45,-0.43]×[0.51,0.54]×[-0.3,-0.28]` 的 16 个角、中心、两个内部插值点，以及零向量、全 1、全 −1。这说明旧 ONNX 与官方 `.mat` 代表相同的原始网络参数和激活，不说明旧执行器的 **plant 缩放** 与 2026 合同相同，也不替代区间网络计算证明。

独立生成的两个原始网络 ONNX 经 ONNX checker 和 ONNX Runtime 22 点比对通过。在服务器既有 `crownreach28` 环境中，`onnx2pytorch.ConvertModel(..., experimental=False).to(torch.float64)` 对两份原始 ONNX 及一份人工激活/仿射测试模型均成功导入，初始盒中心输出分别为 `0.461650623936015`、`-0.0473665577828762`、`2.694999219147924`。第三份是刻意选用 `sigmoid,sigmoid,sigmoid,tanh`、`u=2f+3` 的**构造器自测**，不是任何基准的控制器合同。

本地收据和预检模型：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/tora_reach_controller_preflight_001/`，主收据 `AUDIT.json` 包含逐层布尔结果、22 个输入点和最大前向误差，另外三份 `.onnx.json` 记载独立构造模型的显式激活/缩放与点误差。服务器隔离目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_controller_preflight_001/`。构造与审计入口：[`tools/archcomp26_tora_reach_controller_nohash.py`](../tools/archcomp26_tora_reach_controller_nohash.py)。这些操作没有调用内容摘要或哈希检查。

## 后续按合同生成

工具的 `build` 子命令从指定官方 `.mat` 读取权重，要求调用者**显式**给出四层激活（`mat` 表示采用文件内序列，或逗号分隔四层名称）、控制 `scale` 与 `offset`，输出含 `u=scale·f(x)+offset` 的 float64 ONNX，并立即做 22 点独立数值比较。该 ONNX 的输出已是 plant 输入 `u`；接入四种方法时必须取消执行器现有的二次缩放。工具不提供会静默决定论文/官方/旧合同的默认值。

仍待用户选定的是 sigmoid 版输出激活与缩放（论文叙述、官方 `.mat/.txt`、旧执行器三者不同），以及 ReLU/tanh 版是否遵循论文文字所称 sigmoid 隐层或官方 `.mat` 的 ReLU 隐层。目标 checker 的“时间窗内存在某时刻”与“终点包含于目标”的语义也需要明确映射。预检后没有因缺少 ONNX 文件或科学 Python 包而受阻；主合同未确定前两个 reach 实例保持未尝试。
