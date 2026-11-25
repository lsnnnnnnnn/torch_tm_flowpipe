# Taylor-Model Reachability for Neural-Network-Controlled Systems

**[Research slides (PDF)](presentation/slides.pdf)** · [Editable slides (TeX)](presentation/slides.tex) · [Speaker notes (Word)](presentation/speaker_script.docx)

This project computes bounds on the states a dynamical system can reach under neural feedback. A Taylor model carries a polynomial in the uncertain inputs along with an interval for the remaining error. The slides use worked examples to compare our implementation with Huan Zhang’s benchmark implementation. The corresponding code covers [controller coefficient rounding](method/controllers/strict_injection.py), [controller residual bounds](method/controllers/residual.py), [the independent controller bound](method/controllers/anchored_control.py), [validation within a time step](method/flowstar_gpu/weighted_validation.py), and [the propagation of earlier errors](method/flowstar_gpu/symbolic_remainder.py).

## Run the numerical checks

With Python 3.10 or later, run these commands from the repository root. The component check uses PyTorch on the CPU.

```sh
python -m pip install -e '.[method,test]'
python method/check_components.py
python examples/affine_controlled.py
python -m pytest -q
```

The component check evaluates a neural controller on a partition of 256 subboxes and transports an earlier error through an interval factor. The example advances a polynomial plant with uncertain affine control. The [plant package](src/torch_tm_flowpipe/) provides the Taylor-model primitives used in the example. The [research implementation](method/) contains the controller refinements and the sparse reachability engine used in the comparison. See the [engine attribution and license](method/ATTRIBUTION.md).

## Benchmark results

The [reference driver](method/crown_reach.py) accepts a benchmark configuration. Run `python method/crown_reach.py --help` to inspect its options. Its neural bound path requires ONNX, onnx2pytorch and a compatible auto_LiRPA installation. For CUDA execution, install CUDA-enabled PyTorch with its matching toolkit, C++ compiler and Ninja.

The [benchmark catalog](benchmarks/README.md) identifies the systems and model assets. The [comparison data](results/benchmarks.json) cover 24 tasks and 114 physical state entries, including selected numerical settings, completion status, timing samples, endpoint widths and maximum trajectory widths. For the less-robust Double Pendulum, the selected run reduced the second joint's endpoint angular-velocity width from 1.10979 to 0.58085 radians per second. Its median driver time was 4.79 seconds, compared with 3.51 seconds for Huan's run.

The benchmark measurements use the recorded controllers, configurations, native extensions and device environment. Replaying a complete GPU run requires its matching driver and neural-verification dependencies.
