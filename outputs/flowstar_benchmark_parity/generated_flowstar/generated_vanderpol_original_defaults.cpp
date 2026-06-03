#include "Continuous.h"
#include <ctime>
#include <cstdio>
#include <vector>

// Build/link parity check: compile this file against flowstar-toolbox with -lflowstar.
// Original benchmark parity: default adaptive stepsize is set explicitly below.
// Fixed-step harnesses in this repository use setting.setFixedStepsize(...); this one does not.

using namespace flowstar;
using namespace std;

int main()
{
  Variables vars;
  int x_id = vars.declareVar("x");
  int y_id = vars.declareVar("y");
  int t_id = vars.declareVar("t");

  ODE<Real> ode({"y", "(1 - x^2) * y - x", "1"}, vars);

  Computational_Setting setting(vars);
  setting.setAdaptiveStepsize(0.002, 0.10000000000000001, 4);
  setting.setCutoffThreshold(1e-10);
  Interval remainder(-0.0001, 0.0001);
  vector<Interval> remainder_estimation(vars.size(), remainder);
  setting.setRemainderEstimation(remainder_estimation);

  Interval init_x(1.1000000000000001, 1.3999999999999999);
  Interval init_y(2.3500000000000001, 2.4500000000000002);
  vector<Interval> box(vars.size());
  box[x_id] = init_x;
  box[y_id] = init_y;
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);

  vector<Constraint> safeSet = {Constraint("y - 2.75", vars)};
  Result_of_Reachability result;

  clock_t begin, end;
  begin = clock();
  Symbolic_Remainder sr(initialSet, 100);
  ode.reach(result, initialSet, 10, setting, safeSet, sr);
  end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\n", result.isCompleted() ? 1 : 0);
  printf("FLOWSTAR_SAFE %d\n", result.isSafe() ? 1 : 0);
  printf("FLOWSTAR_UNSAFE %d\n", result.isUnsafe() ? 1 : 0);

  if(!result.isCompleted())
  {
    printf("Flowpipe computation is terminated due to the large overestimation.\n");
  }

  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "generated_vanderpol_t_x", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT generated_vanderpol_t_x t x\n");
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "generated_vanderpol_t_y", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT generated_vanderpol_t_y t y\n");

  return 0;
}
