#define protected public
#include "Continuous.h"
#undef protected
#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(
    const Interval &time_domain, unsigned int order) {
  Real h;
  time_domain.sup(h);
  vector<Real> powers(order + 1, 1);
  for(unsigned int i = 1; i <= order; ++i)
    powers[i] = powers[i-1] * h;
  return powers;
}

static void print_terms(
    const char *kind, const TaylorModelVec<Real> &tmv) {
  for(unsigned int state = 0; state < tmv.tms.size(); ++state) {
    for(list<Term<Real> >::const_iterator term =
            tmv.tms[state].expansion.terms.begin();
        term != tmv.tms[state].expansion.terms.end(); ++term) {
      Real coefficient;
      vector<unsigned int> degrees;
      coefficient = term->coefficient;
      degrees = term->degrees;
      printf("FS_TERM kind=%s state=%u coefficient=%.17g exponents=",
             kind, state, coefficient.toDouble());
      for(unsigned int i = 0; i < degrees.size(); ++i)
        printf("%s%u", i ? "," : "", degrees[i]);
      printf("\n");
    }
    printf(
        "FS_REMAINDER kind=%s state=%u lower=%.17g upper=%.17g\n",
        kind, state, tmv.tms[state].remainder.inf(),
        tmv.tms[state].remainder.sup());
  }
}

static void print_box(
    const char *kind, const TaylorModelVec<Real> &tmv,
    const vector<Interval> &domain) {
  vector<Interval> box;
  tmv.intEval(box, domain);
  for(unsigned int state = 0; state < box.size(); ++state)
    printf("FS_BOX kind=%s state=%u lower=%.17g upper=%.17g\n",
           kind, state, box[state].inf(), box[state].sup());
}

static void print_sample(
    const char *kind, unsigned int sample,
    const TaylorModelVec<Real> &tmv, const vector<Interval> &domain,
    bool endpoint) {
  vector<Interval> point = domain;
  for(unsigned int i = 0; i < point.size(); ++i) {
    double value;
    if(endpoint && i == 0)
      value = 0.0;
    else if(sample == 0)
      value = point[i].inf();
    else if(sample == 1)
      value = 0.5 * (point[i].inf() + point[i].sup());
    else
      value = point[i].sup();
    point[i] = Interval(value);
  }
  vector<Interval> box;
  tmv.intEval(box, point);
  for(unsigned int state = 0; state < box.size(); ++state) {
    printf("FS_SAMPLE kind=%s sample=%u state=%u lower=%.17g upper=%.17g point=",
           kind, sample, state, box[state].inf(), box[state].sup());
    for(unsigned int i = endpoint ? 1 : 0; i < point.size(); ++i)
      printf("%s%.17g", i > (endpoint ? 1u : 0u) ? "," : "", point[i].inf());
    printf("\n");
  }
}

int main() {
  intervalNumPrecision = 53;
  setenv("FLOWSTAR_AUDIT_TRACE", "1", 1);
  unsetenv("FLOWSTAR_AUDIT_DISABLE_REFINEMENT");
  unsetenv("FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT");
  unsetenv("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION");

  setenv("FLOWSTAR_AUDIT_STEP", "1", 1);
  Variables vars;
  int state_0_id = vars.declareVar("x");
  int state_1_id = vars.declareVar("y");
  ODE<Real> ode({"y", "-x"}, vars);
  Computational_Setting setting(vars);
  if(!setting.setFixedStepsize(0.01, 4)) return 3;
  setting.setCutoffThreshold(1.0000000000000001e-15);
  setting.setRemainderEstimation(
      vector<Interval>(vars.size(),
        Interval(-0.001, 0.001)));
  setting.printOff();
  vector<Interval> initial_box(vars.size());
  initial_box[state_0_id] = Interval(0.90000000000000002, 1.1000000000000001);
  initial_box[state_1_id] = Interval(-0.10000000000000001, 0.10000000000000001);
  Flowpipe current(initial_box);
  Flowpipe next;
  vector<Constraint> invariant;
  clock_t begin = clock();
  int advanced = current.advance(
      next, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
  clock_t end = clock();
  printf("FS_STATUS advanced=%d seconds=%.17g variant=flowstar_stock\n",
         advanced, (double)(end - begin) / CLOCKS_PER_SEC);
  if(advanced != 1) return 4;

  TaylorModelVec<Real> composed;
  next.compose(composed, 4, setting.tm_setting.cutoff_threshold);
  for(unsigned int i = 0; i < next.domain.size(); ++i)
    printf("FS_DOMAIN index=%u lower=%.17g upper=%.17g\n",
           i, next.domain[i].inf(), next.domain[i].sup());
  print_terms("tube", composed);
  print_box("tube", composed, next.domain);

  TaylorModelVec<Real> endpoint;
  composed.evaluate_time(
      endpoint, endpoint_powers(next.domain[0], 4));
  vector<Interval> endpoint_domain = next.domain;
  endpoint_domain[0] = Interval(0.0);
  vector<Interval> collapsed_endpoint_box;
  endpoint.intEval(collapsed_endpoint_box, endpoint_domain);
  vector<Interval> native_endpoint_domain = next.domain;
  Real accepted_step;
  next.domain[0].sup(accepted_step);
  native_endpoint_domain[0] = Interval(accepted_step);
  vector<Interval> native_endpoint_box;
  composed.intEval(native_endpoint_box, native_endpoint_domain);
  for(unsigned int state = 0; state < endpoint.tms.size(); ++state) {
    double repaired_lower = collapsed_endpoint_box[state].inf();
    double repaired_upper = collapsed_endpoint_box[state].sup();
    if(native_endpoint_box[state].inf() < repaired_lower)
      repaired_lower = native_endpoint_box[state].inf();
    if(native_endpoint_box[state].sup() > repaired_upper)
      repaired_upper = native_endpoint_box[state].sup();
    double padding_lower =
        repaired_lower - collapsed_endpoint_box[state].inf();
    double padding_upper =
        repaired_upper - collapsed_endpoint_box[state].sup();
    printf(
        "FS_ENDPOINT_PATH state=%u "
        "collapsed_lower=%.17g collapsed_upper=%.17g "
        "native_lower=%.17g native_upper=%.17g "
        "repaired_lower=%.17g repaired_upper=%.17g "
        "padding_lower=%.17g padding_upper=%.17g\n",
        state, collapsed_endpoint_box[state].inf(),
        collapsed_endpoint_box[state].sup(),
        native_endpoint_box[state].inf(), native_endpoint_box[state].sup(),
        repaired_lower, repaired_upper, padding_lower, padding_upper);
  }
  print_terms("collapsed", endpoint);
  print_box("collapsed", endpoint, endpoint_domain);
  for(unsigned int sample = 0; sample < 3; ++sample) {
    print_sample("tube", sample, composed, next.domain, false);
    print_sample("collapsed", sample, endpoint, endpoint_domain, true);
  }
  return 0;
}
