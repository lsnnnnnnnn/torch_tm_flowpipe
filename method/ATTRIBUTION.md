# Source attribution

The `flowstar_gpu` engine derives from Huan Zhang's [flowstar-gpu](https://github.com/huanzhang12/flowstar-gpu), commit `a3fb2e94ba976aaf498c4a9cb3f98165cddcc272`. The upstream package identifies Huan Zhang as the author and declares GPL-3.0-only. The [license text](COPYING.GPL-3.0) accompanies the derivative.

This project adapted the engine and added controller injection, residual refinement, an independent controller bound, time-weighted validation and interval transport of retained errors. The project version of `flowstar_gpu/weighted_validation.py` includes the statistics controls used in the October 2026 experiments.

`crown_reach.py` is the shared reference driver from Xiangru's [CROWN-Reach_Development](https://github.com/xiangruzh/CROWN-Reach_Development), branch `2026_experiment`, commit `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3`. Its SHA-256 digest is `1bc3aeea0e9216fe8432ae78f18c9592d19658c23cdde02b90a54d0b14c08a31`. The October experiments also used benchmark-specific drivers that connected the controller and validation components to the engine.

## Double pendulum controller requirements

`controllers/residual.py` evaluates the double pendulum network with four inputs, two hidden layers of 25 rectified linear units each, and two outputs. It runs the tensor calculations directly on the CPU and on CUDA batches other than 14,400 rows.

In the full CUDA experiment, each of 225 initial boxes contributes 64 subboxes per batch. For these 14,400 rows, the evaluator uses `flowstar_gpu.graphing.GraphCache` to replay the tensor calculations. The first call checks that the fused CUDA extension loaded successfully and raises an error if it is unavailable. Building the extension requires CUDA-enabled PyTorch, a compatible CUDA toolkit and C++ compiler, and Ninja.

`controllers/partition_core.py` connects this evaluator to the original double pendulum controller callbacks and checks for PyTorch 2.5.1. Its optional validation mode uses 225 four-dimensional input boxes on CUDA. Run `python method/check_components.py` from the repository root to exercise the residual, splitting, and error transport on a small CPU example.
