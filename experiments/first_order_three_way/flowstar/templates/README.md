Generated C++ programs use the in-code template in `run_flowstar.py`.  The
template deliberately checks the Boolean return from
`Computational_Setting::setFixedStepsize`; this is how the benchmark records
the installed toolbox's rejection of fixed order 1 instead of silently
continuing with its constructor's adaptive-order-4 defaults.
