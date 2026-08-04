#include "Continuous.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace flowstar;
using namespace std;

int main(const int argc, char ** argv)
{
	if(argc != 7)
	{
		fprintf(stderr, "usage: %s h order x0_lower x0_upper candidate cutoff\n", argv[0]);
		return 2;
	}
	const double h = strtod(argv[1], NULL);
	const unsigned int order = static_cast<unsigned int>(strtoul(argv[2], NULL, 10));
	const double x0_lower = strtod(argv[3], NULL);
	const double x0_upper = strtod(argv[4], NULL);
	const double candidate = strtod(argv[5], NULL);
	const double cutoff = strtod(argv[6], NULL);
	intervalNumPrecision = 53;

	printf(
		"TRACE_CONFIG route=official-stock-native-api h=%.17g order=%u "
		"effective_rhs_order=%u x0_lower=%.17g x0_upper=%.17g "
		"candidate=%.17g cutoff=%.17g preconditioning=diagonal_scaling "
		"symbolic_remainder=disabled precision_bits=53\n",
		h, order, order, x0_lower, x0_upper, candidate, cutoff);

	Variables vars;
	const int x_id = vars.declareVar("x");
	ODE<Real> ode({"1 + 2*x"}, vars);
	Computational_Setting setting(vars);
	if(!setting.setFixedStepsize(h, order))
	{
		return 3;
	}
	setting.setCutoffThreshold(cutoff);
	setting.setRemainderEstimation(
		vector<Interval>(vars.size(), Interval(-candidate, candidate)));
	setting.printOff();

	vector<Interval> initial_box(vars.size());
	initial_box[x_id] = Interval(x0_lower, x0_upper);
	Flowpipe initial_set(initial_box);
	vector<Constraint> safe_set;
	Result_of_Reachability result;
	ode.reach(result, initial_set, h, setting, safe_set);

	printf(
		"TRACE_OFFICIAL status=%d completed=%d safe=%d segments=%zu\n",
		result.status, result.isCompleted() ? 1 : 0,
		result.isSafe() ? 1 : 0, result.flowpipes.size());
	if(!result.isCompleted() || result.flowpipes.empty())
	{
		return 4;
	}

	const Flowpipe & accepted = result.flowpipes.back();
	for(unsigned int i = 0; i < accepted.domain.size(); ++i)
	{
		printf(
			"TRACE_DOMAIN stage=official_accepted index=%u lower=%.17g upper=%.17g\n",
			i, accepted.domain[i].inf(), accepted.domain[i].sup());
	}

	TaylorModelVec<Real> composed;
	accepted.compose(composed, order, setting.tm_setting.cutoff_threshold);
	vector<Interval> tube;
	composed.intEval(tube, accepted.domain);
	Real accepted_h;
	accepted.domain[0].sup(accepted_h);
	vector<Interval> endpoint_domain = accepted.domain;
	endpoint_domain[0] = Interval(accepted_h);
	vector<Interval> endpoint;
	composed.intEval(endpoint, endpoint_domain);
	for(unsigned int i = 0; i < tube.size(); ++i)
	{
		printf(
			"TRACE_BOX stage=official_full_tube scope=full_initial_interval "
			"state=%u lower=%.17g upper=%.17g\n",
			i, tube[i].inf(), tube[i].sup());
		printf(
			"TRACE_BOX stage=official_accepted_right_endpoint scope=full_initial_interval "
			"state=%u lower=%.17g upper=%.17g\n",
			i, endpoint[i].inf(), endpoint[i].sup());
	}
	return 0;
}
