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
  setting.setFixedStepsize(0.0036197592495462228, 6);
  setting.setCutoffThreshold(1e-10);
  vector<Interval> remainder_estimation(vars.size());
  for(unsigned int i = 0; i < vars.size(); ++i)
  {
    remainder_estimation[i] = Interval(-0.0001, 0.0001);
  }
  setting.setRemainderEstimation(remainder_estimation);
  setting.printOn();

  vector<Interval> box(vars.size());
  box[x_id] = Interval(-0.65444411959802862, 4.6073508088899455);
  box[y_id] = Interval(-11.123759781885681, 10.530997883293862);
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);
  vector<Constraint> safeSet;
  Result_of_Reachability result;

  clock_t begin, end;
  begin = clock();
  ode.reach(result, initialSet, 0.0036197592495462228, setting, safeSet);
  end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\n", result.isCompleted() ? 1 : 0);
  printf("FLOWSTAR_SAFE %d\n", result.isSafe() ? 1 : 0);
  printf("FLOWSTAR_UNSAFE %d\n", result.isUnsafe() ? 1 : 0);

  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "oracle_flowstar_o6_t_x", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT oracle_flowstar_o6_t_x t x\n");
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "oracle_flowstar_o6_t_y", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT oracle_flowstar_o6_t_y t y\n");
  return 0;
}
