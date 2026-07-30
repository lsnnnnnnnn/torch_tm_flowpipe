#include "Continuous.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(const Interval &time_domain, unsigned int order) {
  Real h;
  time_domain.sup(h);
  vector<Real> powers(order + 1, 1);
  for(unsigned int i = 1; i <= order; ++i) powers[i] = powers[i-1] * h;
  return powers;
}

static void print_tmv(
    unsigned int step,
    double absolute_time,
    const char *kind,
    const TaylorModelVec<Real> &tmv,
    const vector<Interval> &domain,
    const vector<double> &native_widths,
    const vector<double> &post_widths) {
  vector<Interval> box;
  tmv.intEval(box, domain);
  for(unsigned int state = 0; state < tmv.tms.size(); ++state) {
    Interval polynomial_range;
    tmv.tms[state].polyRange(polynomial_range, domain);
    const Interval &remainder = tmv.tms[state].remainder;
    printf(
        "REPAIR_ROW step=%u time=%.17g kind=%s state=%u "
        "lower=%.17g upper=%.17g poly_width=%.17g remainder_width=%.17g "
        "native_remainder_width=%.17g postprocessed_remainder_width=%.17g\n",
        step, absolute_time, kind, state, box[state].inf(), box[state].sup(),
        polynomial_range.sup() - polynomial_range.inf(),
        remainder.sup() - remainder.inf(), native_widths[state],
        post_widths[state]);
  }
}

int main() {
  intervalNumPrecision = 53;
  Variables vars;
  int state_0_id = vars.declareVar("x1");
  int state_1_id = vars.declareVar("x2");
  ODE<Real> ode({"x1*x2", "x1^2 - x2"}, vars);
  Computational_Setting setting(vars);
  bool accepted = setting.setFixedStepsize(0.01, 4);
  printf("REPAIR_ORDER requested=4 accepted=%d\n", accepted ? 1 : 0);
  if(!accepted) return 3;
  setting.setCutoffThreshold(1.0000000000000001e-15);
  vector<Interval> estimates(
      vars.size(), Interval(-0.0001, 0.0001));
  setting.setRemainderEstimation(estimates);
  setting.printOff();
  vector<Constraint> invariant;
  vector<Interval> initial_box(vars.size());
  initial_box[state_0_id] = Interval(0.080000000000000002, 0.12);
  initial_box[state_1_id] = Interval(0.17999999999999999, 0.22);
  Flowpipe current(initial_box);
  unsigned int steps = (unsigned int)floor(
      0.25 / 0.01 + 0.5);
  double absolute_time = 0.0;
  for(unsigned int step = 1; step <= steps; ++step) {
    char step_buffer[32];
    snprintf(step_buffer, sizeof(step_buffer), "%u", step);
    setenv("FLOWSTAR_AUDIT_STEP", step_buffer, 1);
    clock_t begin = clock();
    Flowpipe next;
    int advanced = current.advance(
        next, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
    clock_t end = clock();
    printf("REPAIR_STEP step=%u code=%d seconds=%.17g\n", step, advanced,
           (double)(end - begin) / CLOCKS_PER_SEC);
    if(advanced != 1) break;
    vector<double> native_widths(next.tmvPre.tms.size(), 0.0);
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      native_widths[state] =
          next.tmvPre.tms[state].remainder.sup() -
          next.tmvPre.tms[state].remainder.inf();

    vector<double> post_widths(next.tmvPre.tms.size(), 0.0);
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      post_widths[state] =
          next.tmvPre.tms[state].remainder.sup() -
          next.tmvPre.tms[state].remainder.inf();
    absolute_time += next.domain[0].sup();
    TaylorModelVec<Real> composed;
    next.compose(composed, 4, setting.tm_setting.cutoff_threshold);
    print_tmv(step, absolute_time, "tube", composed, next.domain,
              native_widths, post_widths);
    TaylorModelVec<Real> endpoint;
    composed.evaluate_time(endpoint, endpoint_powers(next.domain[0], 4));
    vector<Interval> endpoint_domain = next.domain;
    endpoint_domain[0] = Interval(0.0);
    print_tmv(step, absolute_time, "endpoint_raw", endpoint, endpoint_domain,
              native_widths, post_widths);
    vector<Interval> endpoint_box;
    endpoint.intEval(endpoint_box, endpoint_domain);
    current = next;
  }
  return 0;
}
