本轮因第 2.2 节的参考端点漏包在 profiling 前停止，因此没有优化后的剩余时间分布。
这张图只复用 4939fb 已冻结完整运行的 solver / independent export 两项时间；
它不是当前内部热点分解，也不支持 prepared replay 的 f、s 或提速估计。
两项互斥；进程启动没有并入求解时间。原始来源见 full_timings.csv。
