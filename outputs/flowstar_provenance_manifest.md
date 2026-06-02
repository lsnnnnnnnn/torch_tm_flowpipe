# Flow* Provenance Manifest

Scope: plant-only fixed-step/fixed-order comparison against the `chenxin415/flowstar` C++ toolbox/static library backend. This is not a Python Flow* package workflow, not `Flow*_adaptive`, not a full CROWN-Reach NNCS pipeline, and not a raw Taylor-model coefficient comparison.

## torch_tm_flowpipe

- `path`: `/srv/local/shengenli/torch_tm_flowpipe`
- `branch`: `main`
- `head_sha`: `ad8f748b533ef482c680e441c900004c67cd06c8`
- `commit_message`: `ad8f748 2026-06-02-output-manifest-and-flowstar-doc-fix`
- `status_short_before_manifest_generation`: `?? scripts/generate_flowstar_provenance.py`
- `remote_origin_main`: `ad8f748b533ef482c680e441c900004c67cd06c8	refs/heads/main`

## Flow* Backend

- `FLOWSTAR_ROOT`: `/srv/local/shengenli/flowstar`
- backend: `toolbox_cpp`
- Flow* HEAD: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- `libflowstar.a`: `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a`
- `libflowstar.a sha256`: `fdc4f1645d9418c1ab2839e4a821b2065015d30b5f9384ffea5145d3b7afe597`
- compiler: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`

Flow* remotes:

```text
origin	https://github.com/chenxin415/flowstar (fetch)
origin	https://github.com/chenxin415/flowstar (push)
```

Flow* `git status --short`:

```text
?? benchmarks/continuous/simple/simple
?? benchmarks/continuous/simple/simple.o
?? benchmarks/continuous/simple/simple.plt
?? benchmarks/continuous/vanderpol/vanderpol
?? benchmarks/continuous/vanderpol/vanderpol.o
?? benchmarks/continuous/vanderpol/vanderpol_t_x.eps
?? benchmarks/continuous/vanderpol/vanderpol_t_x.plt
?? benchmarks/continuous/vanderpol/vanderpol_t_y.eps
?? benchmarks/continuous/vanderpol/vanderpol_t_y.plt
?? flowstar-2.1.0/__MACOSX/
?? flowstar-2.1.0/flowstar-2.1.0/
?? flowstar-toolbox/Constraints.o
?? flowstar-toolbox/Continuous.o
?? flowstar-toolbox/Discrete.o
?? flowstar-toolbox/Geometry.o
?? flowstar-toolbox/Hybrid.o
?? flowstar-toolbox/Interval.o
?? flowstar-toolbox/Matrix.o
?? flowstar-toolbox/Variables.o
?? flowstar-toolbox/lex.yy.c
?? flowstar-toolbox/lex.yy.o
?? flowstar-toolbox/libflowstar.a
?? flowstar-toolbox/modelParser.output
?? flowstar-toolbox/modelParser.tab.c
?? flowstar-toolbox/modelParser.tab.h
?? flowstar-toolbox/modelParser.tab.o
?? flowstar-toolbox/settings.o
```

## Generated C++ Audit

- Generated C++ cases from CSV: `252`
- Existing C++ case files: `252`
- All cases include `Continuous.h`: `True`
- All cases call `ode.reach(...)`: `True`
- All cases use fixed step/order via `setFixedStepsize(h, order)`: `True`
- Runner links Flow* static library: `True` (`-L $FLOWSTAR_ROOT/flowstar-toolbox -lflowstar`)

## Result Counts

| status | count |
| --- | ---: |
| completed | 122 |
| failed | 130 |

## Box, Ratio, And Runtime Semantics

- Box source: GNUPLOT interval plot files emitted by `plot_2D_interval_GNUPLOT`.
- Flow* GNUPLOT-derived rows have `endpoint_box_available=false`; endpoint ratios are not allowed.
- Current torch-vs-Flow* ratios are `last_segment` and `tube` only.
- Flow* `runtime_s` uses `FLOWSTAR_RUNTIME_S` / internal reach time when present.
- Compile/run wall times are separate: `flowstar_wall_compile_s`, `flowstar_wall_run_s`, `flowstar_wall_total_s`.

## Representative Generated Artifacts

| label | setting | h | steps | order | status | cpp sha256 | stdout sha256 | stderr sha256 | plot paths |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| order2_loose_failed | rem1e-4_cut1e-10 | 0.01 | 10 | 2 | failed | eb5db7abaac4bced9025417ee1b4dffbbb353f672355b61819af9fff405884c3 | 32dc31333739b28733baeec2c7388676cb87d6c2024099a0f02926b07304f75c | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | /srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-4_cut1e-10/van_der_pol_h0_01_s10_o2_t_x.plt (27fbd1bde110d529a473601ec40572f376a01da62cb51f30dec56f073481763d)<br>/srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-4_cut1e-10/van_der_pol_h0_01_s10_o2_t_y.plt (d1432c7f471ce1cda03fa4d9043f5dcc7cc9b21efa191f4e1e318ea3234832f2) |
| order4_loose_completed | rem1e-4_cut1e-10 | 0.01 | 10 | 4 | completed | 16188e0bc8edb3d08beb761d6cfee7e21fae36829011b037199fe648198d4805 | 49a08a1f840edfb4cf5cc61e1612b145f1433267bba5f369f9d903aa37a5870d | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | /srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-4_cut1e-10/van_der_pol_h0_01_s10_o4_t_x.plt (527353154f123bd4a5687a121496b326194922e2483bf0cbf816eb75949625e6)<br>/srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-4_cut1e-10/van_der_pol_h0_01_s10_o4_t_y.plt (fd2754487f51b272842c1db6614b1b29cc2e331fc6df1fb1a5363d875ce5c60f) |
| order8_strict_completed | rem1e-10_cut1e-15 | 0.0025 | 10 | 8 | completed | 7dfb5a68d7e55a963809fa0447f80400efbca51dc310c1023b5e5ebc5f20a3c6 | d6f4b83859f7722be7562e979a4709a803903bb54a2d88706b1661be6eba9fa8 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | /srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-10_cut1e-15/van_der_pol_h0_0025_s10_o8_t_x.plt (4a43a961b283861893ef4ddb7769de9807dfaf218216f0ba4ec28e0cf26daaa8)<br>/srv/local/shengenli/torch_tm_flowpipe/outputs/flowstar_models_rem1e-10_cut1e-15/van_der_pol_h0_0025_s10_o8_t_y.plt (e6909d267d66f4ee7293e7c0b0593de837e2c14f766d18048bf6868acca8ab55) |

## Claim Boundaries

- Plant-only polynomial ODE reachability only.
- Fixed-step/fixed-order baseline only; this does not represent Flow* adaptive or best-tuned performance.
- No endpoint ratio is reported because Flow* endpoint boxes are unavailable from GNUPLOT artifacts.
- No raw Taylor-model coefficient comparison.
- No CROWN, auto_LiRPA, Jacobian bounds, sin/cos, hybrid automata, Flow* core binding, or full CROWN-Reach NNCS pipeline reproduction.
