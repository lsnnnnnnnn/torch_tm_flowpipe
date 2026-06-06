#include "Continuous.h"
#include <ctime>
#include <cstdio>
#include <vector>
using namespace flowstar;
using namespace std;

int main()
{
  Variables vars;
  int x_id = vars.declareVar("x");
  int y_id = vars.declareVar("y");
  int t_id = vars.declareVar("t");

  ODE<Real> ode({"y", "y - x - x^2*y", "1"}, vars);
  Computational_Setting setting(vars);
  setting.setFixedStepsize(0.0023437500000000003, 4);
  setting.setCutoffThreshold(1e-10);
  vector<Interval> remainder_estimation(vars.size());
  for(unsigned int i = 0; i < vars.size(); ++i)
  {
    remainder_estimation[i] = Interval(-0.0001, 0.0001);
  }
  setting.setRemainderEstimation(remainder_estimation);
  setting.printOn();

  vector<Interval> box(vars.size());
  box[x_id] = Interval(-1.4520682697687954, -1.2638150421613479);
  box[y_id] = Interval(-2.4735145668392469, -2.1934891997657759);
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);
  vector<Constraint> safeSet;
  Result_of_Reachability result;

  clock_t begin, end;
  begin = clock();
  ode.reach(result, initialSet, 0.0023437500000000003, setting, safeSet);
  end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\n", result.isCompleted() ? 1 : 0);
  printf("FLOWSTAR_SAFE %d\n", result.isSafe() ? 1 : 0);
  printf("FLOWSTAR_UNSAFE %d\n", result.isUnsafe() ? 1 : 0);

  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "oracle_flowstar_o4_t_x", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT oracle_flowstar_o4_t_x t x\n");
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "oracle_flowstar_o4_t_y", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT oracle_flowstar_o4_t_y t y\n");
  return 0;
}
