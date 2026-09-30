# PyTorch/CUDA 可达域工程阶段汇报

2026-09-23。先读 [详细报告](REPORT.md)，或打开 `report.pdf`。演示直接用 `slides.pdf`；修改 `slides.tex` 可以继续编辑。共29页幻灯片（23页主讲、6页附录），讲稿见 [SPEAKER_NOTES.md](SPEAKER_NOTES.md)。

本次材料解释项目背景、主要工作、进展与剩余差距，并给出我们/Huan/Xiangru/Flow* 的相同实验时间和宽度。原式固定plant、作者NNCS、单次失败实验和历史筛查分别说明，不把不同模式、任务和计时边界混在一起。

## 编译与使用

已有PDF可直接使用，无需服务器。LaTeX源码使用XeLaTeX、ctex、Beamer和TeX Live自带Fandol字体，不依赖本机商业字体。

```sh
sh build.sh
```

也可以上传整个目录到Overleaf，编译器选择 **XeLaTeX**，主文件选择 `slides.tex` 或 `report.tex`。仅编译LaTeX不需要Python。

如需从数据重画图并生成所有源文件：

```sh
python build_slides.py
sh build.sh
```

Python需要NumPy和Matplotlib。`build_slides.py` 调用 `build_materials.py`，核对120个plant原始计时样本、60个计入NNCS时间以及65,600行/80通道plant宽度。手工修改 `slides.tex` 后应直接运行 `build.sh`；再次运行生成脚本会重新生成TeX源文件。

## 数据与实验配置

- `evidence/`：原始时间、统计、审计与源码/模型/实际库身份。
- `tables/`：全轨迹范围CSV（gzip）、汇总表、末步实际宽度；可用常见CSV工具读取。
- `configs/plant_contracts.json`：四个固定plant任务，包括全部32个真实分区、十六进制端点与实际设置。
- `configs/tora.yaml`、`single_pendulum.yaml`：原始作者配置；相应 `*_executed.yaml` 为精确执行文件，哈希匹配本次实验。
- `configs/cartpole.yaml`：本次执行的registry full小初盒配置；不是4096分区或10秒ARCH-COMP任务。
- 模型ONNX权重、大型factored模型流及CUDA二进制保留在服务器；本包不包含它们，也不声称脱离原环境即可重跑求解器。
- `verify_bundle.py`：检查交付文件SHA及主要计时统计、执行配置与模型身份。`SHA256SUMS.json` 是本包文件清单；历史来源索引另在 `evidence/SOURCE_MANIFEST.json`。

主比较版本：plant engine `f53696c8` / adapter `4beba860`；NNCS候选 `fff9d0f5`；Huan `d5f0b68f`；Xiangru `1c16d4ef`。独立性能候选 `d8bdc4d2` 和输出候选 `1bc9b40f` 没有在本资料分支里自动合并为一个新正式引擎。

## 当前决定

按用户要求：**本次文档、演示和新分支推送完成后暂停**。固定原式VDP的时间与宽度、TORA末段范围、严格控制器资格及候选统一入口仍未完成；不将本阶段交付标记为总加速目标完成。没有新的优化或benchmark在本资料整理过程中启动。
