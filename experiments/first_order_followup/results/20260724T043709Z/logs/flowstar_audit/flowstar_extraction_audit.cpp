#include "Continuous.h"
#include <cstdio>
#include <list>
#include <vector>
using namespace flowstar;
using namespace std;

static void print_box(unsigned int step, double absolute_time, const char *path,
                      const vector<Interval> &box) {
  for(unsigned int d = 0; d < box.size(); ++d) {
    printf("AUDIT_ROW %u %.17g %s %u %.17g %.17g\n",
           step, absolute_time, path, d, box[d].inf(), box[d].sup());
  }
}

static void print_domain(const char *layer, unsigned int step,
                         const vector<Interval> &domain) {
  for(unsigned int i = 0; i < domain.size(); ++i) {
    printf("AUDIT_DOMAIN %s %u %u %.17g %.17g\n",
           layer, step, i, domain[i].inf(), domain[i].sup());
  }
}

static void print_tm_decomposition(unsigned int step, double absolute_time,
                                   const char *poly_name, const char *rem_name,
                                   const TaylorModelVec<Real> &tmv,
                                   const vector<Interval> &domain) {
  for(unsigned int d = 0; d < tmv.tms.size(); ++d) {
    Interval polynomial;
    tmv.tms[d].polyRange(polynomial, domain);
    vector<Interval> one(1, polynomial);
    print_box(step, absolute_time, poly_name, one);
    one[0] = tmv.tms[d].remainder;
    print_box(step, absolute_time, rem_name, one);
  }
}

static vector<Real> endpoint_powers(const Interval &time_domain, unsigned int order) {
  Real h;
  time_domain.sup(h);
  vector<Real> powers;
  powers.push_back(1);
  powers.push_back(h);
  Real p = h;
  for(unsigned int k = 2; k <= order; ++k) {
    p *= h;
    powers.push_back(p);
  }
  return powers;
}

static void eval_tm_paths(unsigned int step, double absolute_time,
                          const char *direct_name, const char *sub_name,
                          const TaylorModelVec<Real> &tmv,
                          const vector<Interval> &domain, unsigned int order) {
  vector<Interval> endpoint_domain = domain;
  endpoint_domain[0] = domain[0].sup();
  vector<Interval> direct;
  tmv.intEval(direct, endpoint_domain);
  print_box(step, absolute_time, direct_name, direct);

  TaylorModelVec<Real> endpoint_tm;
  tmv.evaluate_time(endpoint_tm, endpoint_powers(domain[0], order));
  vector<Interval> substituted_domain = domain;
  substituted_domain[0] = Interval(0.0);
  vector<Interval> substituted;
  endpoint_tm.intEval(substituted, substituted_domain);
  print_box(step, absolute_time, sub_name, substituted);
}

int main() {
  Variables vars;
  vars.declareVar("x");
  ODE<Real> ode({"x^2"}, vars);
  Computational_Setting setting(vars);
  const unsigned int order = 2;
  if(!setting.setFixedStepsize(0.02, order)) return 3;
  setting.setCutoffThreshold(1e-15);
  vector<Interval> estimates(
      1, Interval(-0.0001,
                  0.0001));
  setting.setRemainderEstimation(estimates);
  setting.printOff();

  vector<Interval> initial_box(1, Interval(0.0, 0.1));
  Flowpipe initial_set(initial_box);
  Result_of_Reachability raw;
  vector<Constraint> safe;
  ode.reach(raw, initial_set, 0.10000000000000001, setting, safe);
  printf("AUDIT_COMPLETED %d\n", raw.isCompleted() ? 1 : 0);
  printf("AUDIT_RAW_SEGMENTS %u\n", (unsigned int)raw.flowpipes.size());

  Result_of_Reachability transformed = raw;
  unsigned int step = 0;
  double absolute_time = 0.0;
  for(list<Flowpipe>::const_iterator it = raw.flowpipes.begin();
      it != raw.flowpipes.end(); ++it) {
    ++step;
    absolute_time += it->domain[0].sup();
    print_domain("raw", step, it->domain);

    vector<Interval> tmv_pre_range;
    it->tmvPre.intEval(tmv_pre_range, it->domain);
    print_box(step, absolute_time, "raw_tmvPre_domain_eval", tmv_pre_range);
    vector<Interval> tmv_range;
    it->tmv.intEval(tmv_range, it->domain);
    print_box(step, absolute_time, "raw_tmv_domain_eval", tmv_range);

    vector<Interval> official_int_eval;
    it->intEval(official_int_eval, order, setting.tm_setting.cutoff_threshold);
    print_box(step, absolute_time, "raw_Flowpipe_intEval_tube", official_int_eval);

    TaylorModelVec<Real> composed;
    it->compose(composed, order, setting.tm_setting.cutoff_threshold);
    vector<Interval> composed_tube;
    composed.intEval(composed_tube, it->domain);
    print_box(step, absolute_time, "raw_compose_tube", composed_tube);
    print_tm_decomposition(step, absolute_time, "raw_compose_polynomial_tube",
                           "raw_compose_remainder_tube", composed, it->domain);
    eval_tm_paths(step, absolute_time, "raw_compose_endpoint_direct",
                  "raw_compose_endpoint_substitute", composed, it->domain, order);
    vector<Interval> raw_endpoint_domain = it->domain;
    raw_endpoint_domain[0] = it->domain[0].sup();
    print_tm_decomposition(step, absolute_time, "raw_compose_polynomial_endpoint",
                           "raw_compose_remainder_endpoint", composed,
                           raw_endpoint_domain);

    TaylorModelVec<Real> composed_normal;
    it->compose_normal(composed_normal, setting.tm_setting.step_exp_table,
                       order, setting.tm_setting.cutoff_threshold);
    vector<Interval> normal_tube;
    composed_normal.intEval(normal_tube, it->domain);
    print_box(step, absolute_time, "raw_compose_normal_tube", normal_tube);
    eval_tm_paths(step, absolute_time, "raw_compose_normal_endpoint_direct",
                  "raw_compose_normal_endpoint_substitute",
                  composed_normal, it->domain, order);
  }

  transformed.transformToTaylorModels(setting);
  printf("AUDIT_TRANSFORMED_SEGMENTS %u\n",
         (unsigned int)transformed.tmv_flowpipes.size());
  step = 0;
  absolute_time = 0.0;
  for(list<TaylorModelFlowpipe>::const_iterator it =
          transformed.tmv_flowpipes.tmv_flowpipes.begin();
      it != transformed.tmv_flowpipes.tmv_flowpipes.end(); ++it) {
    ++step;
    absolute_time += it->domain[0].sup();
    print_domain("transformed", step, it->domain);
    vector<Interval> tube;
    it->tmv_flowpipe.intEval(tube, it->domain);
    print_box(step, absolute_time, "transformed_tube", tube);
    print_tm_decomposition(step, absolute_time, "transformed_polynomial_tube",
                           "transformed_remainder_tube",
                           it->tmv_flowpipe, it->domain);
    eval_tm_paths(step, absolute_time, "transformed_endpoint_direct",
                  "transformed_endpoint_substitute",
                  it->tmv_flowpipe, it->domain, order);
    vector<Interval> transformed_endpoint_domain = it->domain;
    transformed_endpoint_domain[0] = it->domain[0].sup();
    print_tm_decomposition(step, absolute_time,
                           "transformed_polynomial_endpoint",
                           "transformed_remainder_endpoint",
                           it->tmv_flowpipe, transformed_endpoint_domain);
  }

  // Flow* first proves that the configured candidate remainder maps into
  // itself, then replaces it with a refined image.  The toolbox accepts a
  // later refinement even when that image is no longer self-mapping.  This
  // focused path retains the already-proved candidate remainder instead.
  Flowpipe safe_current(initial_box);
  absolute_time = 0.0;
  const unsigned int requested_steps =
      (unsigned int)floor(0.10000000000000001 / 0.02 + 0.5);
  for(step = 1; step <= requested_steps; ++step) {
    Flowpipe safe_next;
    int advanced = safe_current.advance(
        safe_next, ode.expressions, setting.tm_setting, safe, setting.g_setting);
    printf("AUDIT_SAFE_ADVANCE %u %d\n", step, advanced);
    if(advanced != 1) return 5;
    for(unsigned int d = 0; d < safe_next.tmvPre.tms.size(); ++d) {
      safe_next.tmvPre.tms[d].remainder =
          setting.tm_setting.remainder_estimation[d];
    }
    absolute_time += safe_next.domain[0].sup();
    TaylorModelVec<Real> safe_composed;
    safe_next.compose(
        safe_composed, order, setting.tm_setting.cutoff_threshold);
    vector<Interval> safe_tube;
    safe_composed.intEval(safe_tube, safe_next.domain);
    print_box(step, absolute_time, "safe_candidate_tube", safe_tube);
    eval_tm_paths(step, absolute_time, "safe_candidate_endpoint_direct",
                  "safe_candidate_endpoint_substitute",
                  safe_composed, safe_next.domain, order);
    safe_current = safe_next;
  }
  return raw.isCompleted() ? 0 : 4;
}
