# Limitations

This code is a standalone minimal Taylor-model flowpipe research prototype.
Current limitations are intentional and should be stated clearly in reports:

* only polynomial ODE right-hand sides are supported;
* no `sin`, `cos`, `exp`, `log`, or general transcendental Taylor-model support;
* no Flow* parser or Flow* bindings;
* no hybrid automata modes, guards, jumps, domain contraction, or guard range
  over-approximation;
* no symbolic remainder support;
* no adaptive step-size or adaptive Taylor-model order;
* no CROWN, auto_LiRPA, CROWN-Reach, branch-and-bound, or Jacobian/sensitivity
  bound integration;
* CUDA is only a tensor-backend smoke path here and should not be presented as a
  sparse-polynomial GPU optimization;
* floating-point soundness is prototype-level only.  The implementation uses
  outward nudging with `torch.nextafter`, but it is not a production-grade sound
  verifier.

Suggested project wording:

> standalone PyTorch-native minimal Taylor Model flowpipe research prototype for
> polynomial plant ODEs.
