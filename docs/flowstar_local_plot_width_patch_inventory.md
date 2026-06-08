# Flowstar Local Plot/Width Patch Inventory

Flow* checkout path: `/srv/local/shengenli/flowstar`

## Git State

- Branch: `master`
- HEAD: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- Worktree clean: no. There are untracked generated binaries/object files/plot files, but no tracked source diff.
- `git diff --stat`: empty output.

`git status --short --branch` reported:

```text
## master...origin/master
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

`git log --oneline -8` reported:

```text
b85a321 Merge branch 'master' of https://github.com/chenxin415/flowstar
ed08e67 New constructors in TM class is added
6172bcd Update Matrix.h
a49d73f Update README.md
f9855f0 Update README.md
eac80d2 Update spring_pendulum.cpp
f0a915c benchmark updated
bd9c5ad new benchmark added
```

## Plot/Width Search

`rg -n "FLOWSTAR_RUNTIME_S" /srv/local/shengenli/flowstar` found no matches.

Searches for `width_over_time`, `last_segment`, `tube_width`, `segment width`, `plot.*width`, `intermediate.*width`, and `final.*width` found no tracked local source patch that adds width-over-time or final-width plotting. The broad plot/width search found standard Flow* plotting and interval width APIs, including:

- `benchmarks/continuous/vanderpol/vanderpol.cpp:125`: calls `plot_setting.plot_2D_interval_GNUPLOT("./", "vanderpol_t_x", result.tmv_flowpipes, setting)`.
- `benchmarks/continuous/vanderpol/vanderpol.cpp:129`: calls `plot_setting.plot_2D_interval_GNUPLOT("./", "vanderpol_t_y", result.tmv_flowpipes, setting)`.
- `flowstar-toolbox/Continuous.h:2854-2857`: declares GNUPLOT plot helpers.
- `flowstar-toolbox/Continuous.cpp:9569-9585`: dispatches and implements GNUPLOT interval plotting.
- `flowstar-toolbox/Interval.h:218-220` and `flowstar-toolbox/Interval.cpp:1086-1112`: interval width APIs.
- `flowstar-toolbox/Matrix.h:89,926-941`: matrix width API.

Generated Flow* Van der Pol plot artifacts found in the local checkout:

- `benchmarks/continuous/vanderpol/vanderpol_t_x.plt`
- `benchmarks/continuous/vanderpol/vanderpol_t_y.plt`
- `benchmarks/continuous/vanderpol/vanderpol_t_x.eps`
- `benchmarks/continuous/vanderpol/vanderpol_t_y.eps`

Other matching reference/generated paths included:

- `flowstar-2.1.0/flowstar-2.1.0/vanderpol.model`
- `images/benchmarks/vanderpol_t_x.png`
- `images/benchmarks/vanderpol_t_y.png`
- `benchmarks/continuous/vanderpol/vanderpol.cpp`

## Conclusion

No local uncommitted tracked Flow* patch appears to add runtime markers, intermediate width plotting, final width plotting, `width_over_time`, `last_segment`, or `tube_width` instrumentation. The local checkout does contain generated Van der Pol GNUPLOT/EPS artifacts and build products.

Flow* source was inspected locally but not committed into `torch_tm_flowpipe`.
