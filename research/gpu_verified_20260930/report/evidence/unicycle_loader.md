# Unicycle 首步启动失败：环境证据与待办提案

**实际状态：只启动过 ours 一次，在植物计算前退出。Huan、Xiangru 未启动；没有重跑、没有修改算法或重编译缓存。** GPU3 已在 Unix1790682052.40 查询确认为空，交回根代理后续任务。当前按用户“做完这步停一下”要求，仅整理证据，不实现或执行下述提案。

## 唯一实际尝试

原 COMMAND SHA `abaf7bc337a4b05c0db776baf65c01a0754c9087e3c80c05535f6c1ca8e3a8c2`；GPU3 V100、CPU18–21、600秒 watchdog。启动时间 Unix1790681990.172，watch PID1649816、runner PID1649817。启动前122项源/库/模型 SHA、三仓库 HEAD/clean 和资源检查通过。

watch 在3.132135秒后正常观察到 runner rc1，status=error，peak tree RSS572502016 B、GPU322961408 B。两个 owned PID 已退出，不需要清理动作。

原 traceback 在 `run_gpu_suite.py:102`：

```python
assert all([cuda_kernels.available(), tape_kernels.available(), tape_kernels.valid_available()])
```

失败发生在模型运行/植物推进前；没有 DIAGNOSTIC_RESULT、states 或首步 ranges 资格证据。此事应分类为加载环境失败，不能算作数值拒绝、宽度结果或 GPU 单步时间。

## 已证实与未取回的错误

实际冻结 `cuda_kernels.py` 的 `load_cuda_extension`（488–532行）在每个新进程中调用 Torch `load_inline`；它对任何异常都 `except Exception: return None`，没有保存异常内容。因此**原失败进程被吞掉的完整异常无法事后恢复**。原 assert 本身没有指出三个模块分别在哪一步失败。

随后使用 `CUDA_VISIBLE_DEVICES=''` 的独立 CPU 只读诊断，没有调用 `available()`、`load_inline`、编译器构建或植物程序：

- 原环境 PATH 为系统路径，`ninja` 不在 PATH；`sys.executable` 指向 nncs_env/bin/python，该目录也没有 ninja，因此原 loader 的“把 venv/bin 加到 PATH”逻辑帮不上忙。
- 直接调用 Torch 原 `verify_ninja_availability()` 得到确定异常：`RuntimeError('Ninja is required to load C++ extensions')`。这是可直接复现的 JIT 前置环境阻断，符合原 loader 返回 None 的路径；不冒充已经取回原被吞 traceback。
- 实际 CUDA_HOME 环境变量未设，但 Torch 找到 `/usr/local/cuda`，realpath 就是 `/usr/local/cuda-12.6`；两种路径下 nvcc 都为12.6.68，原版本前置检查均通过，仅记录 Torch12.1 vs nvcc12.6 的 minor warning。因此不能把缺少显式 CUDA_HOME 单独定为原因。
- 当前 CC/CXX 未设，系统 `/usr/bin/c++` 是15.2.0，`/usr/bin/g++-13` 不存在。原缓存 build.ninja 则记录 `.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++` 和对应 gcc ccbin；这两个原编译器路径仍存在。恢复 JIT 环境还需考虑这一差别，不能随便让它重建缓存。

最重要的隔离证据：在同一个 CPU 隐藏 GPU 环境，使用 importlib 直接导入原已钉 SHA 的三个现成扩展 **全部成功**，没有改文件，Torch CUDA 初始化状态前后均为 false：

| 扩展 | 原缓存 SHA256 |
|---|---|
| flowstar_seg_kernels | a70749911578252c1e24a2badcdfb35ac5f98b8cf95c3ddd23c486ffb37f124d |
| flowstar_tape_kernels | 83128a2d89334fbf8e4b935f995b3e700f3ed0ee00f942159615ad2cd03b0eff |
| flowstar_valid_kernels | e976569173db6a73fe4cd6b63c8e2d91583db3f99d094be098f020dcd9c11eba |

这证明该三份现成库在当前 Python/Torch 环境下可被动态加载，不证明任何 CUDA kernel 或植物数值步骤已执行。原 LD_LIBRARY_PATH 为空并未妨碍这一实际直接加载。

## 最小待办提案（尚未实现）

下次恢复时，**优先评估还原历史启动环境**这一较简单方案：补回原 PATH 中的 Ninja，明确使用缓存 build.ninja 记录的原 CC/CXX（`.huan-audit-gxx13`），可显式写出同一 CUDA12.6路径；prepared、源代码和原SO全部不变。先确认原缓存能够被原样复用，若需要重建或覆盖任一冻结SO则停止这一方案；不能以“只是修环境”为由让JIT悄悄重编译。当前没有继续查找Ninja位置、测试该恢复环境或重跑。

已完整读取原 `four_way_run_engine_v1.py` 的 prepare：它选择冻结仓库和 TORCH_EXTENSIONS_DIR，没有直接加载现成 SO 的公开参数。若上述原环境无法安全复用缓存，再考虑以下独立直接加载备选；不要修改原 prepared 包、源仓库或现有缓存。

后续如获得新的执行授权，备选是一个独立、短小的 v2 外层入口：

1. 继续核全部 source/prepared/model/SO SHA，保持原 prepared_v1/run_gpu_one_step.py 完全不变。
2. 在该新进程进入原 wrapper 前，将 Torch `load_inline` 的请求名限定到该 arm 历史 `extensions` 清单；仅通过 importlib 读取清单指定的既存路径，读取前后核 SHA，并验证请求的函数在模块中存在。未知名字或缺少库一律拒绝，禁止 JIT fallback。
3. 原 wrapper、原 engine、原 arithmetic 和实际 `.so` 字节全部不变，原 source/binary/endpoint 和固定RPC注入门仍执行；另存 LOADER_RECEIPT，记录适配入口 SHA、每次请求名/实际路径/SHA/导入结果，明确这属于加载适配。
4. 使用新输出目录保留此次失败。先做最小 CPU 导入检查，再待根代理分配空闲 GPU 做原三次首步；不重编译、不扩大数值范围、不跳过原拒绝条件。

不能只补 PATH 后直接不加检查地重跑 `load_inline`：它可能依据当前 CC/CXX、文件时间或编译选项触发缓存重建。原环境方案若不能满足“原历史二进制不变”，再用直接加载的备选。此处仅提出方案，没有新 loader 实现、没有 v2 GPU 执行。

## 已保存的实际证据

`ours_failure_v1/LOCAL_VERIFICATION.json`（SHA `de544af55f3afcd07a98db3a5093a2735207f7a6caf6efd866833c8a2e2416a4`）逐文件绑定8份归档：原 LAUNCH、watch process.json、原 traceback stdout.log、两份实际 loader 源码、三个 CPU 收据。

- `OURS_GPU1STEP_LOADER_CPU_READ_v1.json`：3个现成SO直接import/ldd/build.ninja；SHA `088683e73ba1a07e541dc6d82b6b7c243451486cebec98d9ee33690922a17c70`。
- `OURS_GPU1STEP_CUDA_VERSION_CPU_READ_v2.json`：实际编译器/CUDA路径和版本前置检查；SHA `bbcc815dbbf0ac319521e5a75db010b0340645d5bea89c60f5773d0900d71034`。
- `OURS_GPU1STEP_BUILD_PREREQUISITES_CPU_READ_v1.json`：原 Ninja 检查的实际异常；SHA `b0875e78a25f697b65180610f93a88081e9d0101ca4e3b807d310ad50f83da15`。

收据文件名的 `GPU1STEP` 指“一步”，此次唯一运行使用的是 GPU3，不是 GPU1。早期版本前置探针先假设 `/usr/bin/g++-13` 存在而以 FileNotFoundError 退出；随后v2按原 loader 的实际 fallback 读系统c++。该探针没有编译、没有GPU初始化或实验重跑。

原生 QUAD 的停止仍未获用户授权；本项没有停止其四个进程、没有启动24小时替代作业，也没有在最后一次GPU释放确认之后再次使用GPU3。后续原生/候选作业的最新状态由根代理统一记录。
