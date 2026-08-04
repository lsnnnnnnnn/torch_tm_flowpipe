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

struct Config
{
	double h;
	unsigned int order;
	double x0_lower;
	double x0_upper;
	double candidate;
	double cutoff;
};

static Config parse_config(const int argc, char ** argv)
{
	if(argc != 7)
	{
		fprintf(stderr, "usage: %s h order x0_lower x0_upper candidate cutoff\n", argv[0]);
		exit(2);
	}

	Config config;
	config.h = strtod(argv[1], NULL);
	config.order = static_cast<unsigned int>(strtoul(argv[2], NULL, 10));
	config.x0_lower = strtod(argv[3], NULL);
	config.x0_upper = strtod(argv[4], NULL);
	config.candidate = strtod(argv[5], NULL);
	config.cutoff = strtod(argv[6], NULL);
	return config;
}

static void print_exponents(const vector<unsigned int> & degrees)
{
	for(unsigned int i = 0; i < degrees.size(); ++i)
	{
		printf("%s%u", i == 0 ? "" : ",", degrees[i]);
	}
}

static void print_real_tmv(
	const char * stage, const TaylorModelVec<Real> & tmv,
	const vector<Interval> & domain)
{
	for(unsigned int state = 0; state < tmv.tms.size(); ++state)
	{
		for(list<Term<Real> >::const_iterator term =
				tmv.tms[state].expansion.terms.begin();
			term != tmv.tms[state].expansion.terms.end(); ++term)
		{
			printf(
				"TRACE_TERM stage=%s state=%u coefficient_lower=%.17g "
				"coefficient_upper=%.17g exponents=",
				stage, state, term->coefficient.toDouble(),
				term->coefficient.toDouble());
			print_exponents(term->degrees);
			printf("\n");
		}
		printf(
			"TRACE_REMAINDER stage=%s state=%u lower=%.17g upper=%.17g\n",
			stage, state, tmv.tms[state].remainder.inf(),
			tmv.tms[state].remainder.sup());
	}

	vector<Interval> box;
	tmv.intEval(box, domain);
	for(unsigned int state = 0; state < box.size(); ++state)
	{
		printf(
			"TRACE_BOX stage=%s scope=domain state=%u lower=%.17g upper=%.17g\n",
			stage, state, box[state].inf(), box[state].sup());
	}
}

static void print_interval_tmv(
	const char * stage, const TaylorModelVec<Interval> & tmv,
	const vector<Interval> & domain)
{
	for(unsigned int state = 0; state < tmv.tms.size(); ++state)
	{
		for(list<Term<Interval> >::const_iterator term =
				tmv.tms[state].expansion.terms.begin();
			term != tmv.tms[state].expansion.terms.end(); ++term)
		{
			printf(
				"TRACE_TERM stage=%s state=%u coefficient_lower=%.17g "
				"coefficient_upper=%.17g exponents=",
				stage, state, term->coefficient.inf(), term->coefficient.sup());
			print_exponents(term->degrees);
			printf("\n");
		}
		printf(
			"TRACE_REMAINDER stage=%s state=%u lower=%.17g upper=%.17g\n",
			stage, state, tmv.tms[state].remainder.inf(),
			tmv.tms[state].remainder.sup());
	}

	vector<Interval> box;
	tmv.intEval(box, domain);
	for(unsigned int state = 0; state < box.size(); ++state)
	{
		printf(
			"TRACE_BOX stage=%s scope=domain state=%u lower=%.17g upper=%.17g\n",
			stage, state, box[state].inf(), box[state].sup());
	}
}

template <class DATA_TYPE>
static void print_point_box(
	const char * stage, const char * scope,
	const TaylorModelVec<DATA_TYPE> & tmv, vector<Interval> domain,
	const double t, const double xi)
{
	domain[0] = Interval(t);
	if(domain.size() > 1)
	{
		domain[1] = Interval(xi);
	}
	vector<Interval> box;
	tmv.intEval(box, domain);
	for(unsigned int state = 0; state < box.size(); ++state)
	{
		printf(
			"TRACE_BOX stage=%s scope=%s state=%u lower=%.17g upper=%.17g\n",
			stage, scope, state, box[state].inf(), box[state].sup());
	}
}

static vector<Real> endpoint_powers(
	const Interval & time_domain, const unsigned int order)
{
	Real h;
	time_domain.sup(h);
	vector<Real> powers(order + 1, 1);
	for(unsigned int i = 1; i <= order; ++i)
	{
		powers[i] = powers[i-1] * h;
	}
	return powers;
}

int main(const int argc, char ** argv)
{
	const Config config = parse_config(argc, argv);
	intervalNumPrecision = 53;

	printf(
		"TRACE_CONFIG route=generated-stock h=%.17g order=%u "
		"effective_rhs_order=%u x0_lower=%.17g x0_upper=%.17g "
		"candidate=%.17g cutoff=%.17g preconditioning=diagonal_scaling "
		"symbolic_remainder=disabled precision_bits=53\n",
		config.h, config.order, config.order, config.x0_lower,
		config.x0_upper, config.candidate, config.cutoff);

	Variables vars;
	const int x_id = vars.declareVar("x");
	ODE<Real> ode({"1 + 2*x"}, vars);
	Computational_Setting setting(vars);
	if(!setting.setFixedStepsize(config.h, config.order))
	{
		return 3;
	}
	setting.setCutoffThreshold(config.cutoff);
	setting.setRemainderEstimation(
		vector<Interval>(vars.size(), Interval(-config.candidate, config.candidate)));
	setting.printOff();

	vector<Interval> initial_box(vars.size());
	initial_box[x_id] = Interval(config.x0_lower, config.x0_upper);
	Flowpipe current(initial_box);
	Flowpipe accepted;
	vector<Constraint> invariant;

	print_real_tmv("initial_tmv_pre", current.tmvPre, current.domain);
	print_real_tmv("initial_tmv_right", current.tmv, current.domain);

	const int advanced = current.advance(
		accepted, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
	printf("TRACE_STATUS advanced=%d\n", advanced);
	if(advanced != 1)
	{
		return 4;
	}

	for(unsigned int i = 0; i < accepted.domain.size(); ++i)
	{
		printf(
			"TRACE_DOMAIN stage=accepted index=%u lower=%.17g upper=%.17g\n",
			i, accepted.domain[i].inf(), accepted.domain[i].sup());
	}

	// Read-only diagnostic replay of Continuous.cpp:857-1040.  Every numerical
	// operation below calls the same stock Taylor-model methods used by advance;
	// the accepted object above is never mutated.
	const unsigned int range_dim = ode.expressions.size();
	const unsigned int range_dim_ext = range_dim + 1;
	const Interval int_unit(-1, 1);

	TaylorModelVec<Real> tmv_of_x0;
	current.tmvPre.evaluate_time(tmv_of_x0, setting.tm_setting.step_end_exp_table);
	print_real_tmv("initial_time_evaluated", tmv_of_x0, current.domain);

	vector<Real> const_of_x0;
	tmv_of_x0.constant(const_of_x0);
	TaylorModelVec<Real> tmv_c0(const_of_x0, range_dim_ext);
	tmv_of_x0.rmConstant();

	TaylorModelVec<Real> normalization_map;
	vector<Interval> tmv_poly_range;
	current.tmv.polyRangeNormal(
		tmv_poly_range, setting.tm_setting.step_end_exp_table);
	tmv_of_x0.insert_ctrunc_normal(
		normalization_map, current.tmv, tmv_poly_range,
		setting.tm_setting.step_end_exp_table, current.domain.size(),
		setting.tm_setting.order, setting.tm_setting.cutoff_threshold);

	vector<Interval> range_of_x0;
	normalization_map.intEvalNormal(
		range_of_x0, setting.tm_setting.step_end_exp_table);
	vector<Real> scale;
	vector<Real> inverse_scale;
	for(unsigned int i = 0; i < range_dim; ++i)
	{
		Real magnitude;
		range_of_x0[i].mag(magnitude);
		if(magnitude == 0)
		{
			scale.push_back(0);
			inverse_scale.push_back(1);
		}
		else
		{
			scale.push_back(magnitude);
			inverse_scale.push_back(1 / magnitude);
			range_of_x0[i] = int_unit;
		}
	}
	normalization_map.scale_assign(inverse_scale);
	print_real_tmv("normalization_map", normalization_map, accepted.domain);

	TaylorModelVec<Real> new_x0(scale);
	new_x0 += tmv_c0;
	print_real_tmv("normalized_initial", new_x0, accepted.domain);

	TaylorModelVec<Real> picard = new_x0;
	for(unsigned int i = 1; i <= setting.tm_setting.order; ++i)
	{
		picard.Picard_no_remainder_assign(
			new_x0, ode.expressions, range_dim_ext, i,
			setting.tm_setting.cutoff_threshold);
		const string stage = "polynomial_picard_order_" + to_string(i);
		print_real_tmv(stage.c_str(), picard, accepted.domain);
		print_point_box(
			stage.c_str(), "endpoint_lower_corner", picard, accepted.domain,
			config.h, -1.0);
		print_point_box(
			stage.c_str(), "endpoint_upper_corner", picard, accepted.domain,
			config.h, 1.0);
	}

	for(unsigned int i = 0; i < range_dim; ++i)
	{
		picard.tms[i].remainder = setting.tm_setting.remainder_estimation[i];
	}
	print_real_tmv("candidate_remainder_seed", picard, accepted.domain);
	print_point_box(
		"candidate_remainder_seed", "endpoint_lower_corner", picard,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"candidate_remainder_seed", "endpoint_upper_corner", picard,
		accepted.domain, config.h, 1.0);

	TaylorModelVec<Interval> picard_image;
	list<Interval> intermediate_ranges;
	picard.Picard_ctrunc_normal(
		picard_image, new_x0, ode.expressions,
		setting.tm_setting.step_exp_table, range_dim_ext,
		setting.tm_setting.order, setting.tm_setting.cutoff_threshold,
		intermediate_ranges, setting.g_setting);
	print_interval_tmv("picard_ctrunc_raw", picard_image, accepted.domain);
	print_point_box(
		"picard_ctrunc_raw", "endpoint_lower_corner", picard_image,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"picard_ctrunc_raw", "endpoint_upper_corner", picard_image,
		accepted.domain, config.h, 1.0);

	vector<Interval> roundoff_differences;
	for(unsigned int i = 0; i < range_dim; ++i)
	{
		Polynomial<Interval> difference;
		difference = picard_image.tms[i].expansion - picard.tms[i].expansion;
		Interval range;
		difference.intEvalNormal(range, setting.tm_setting.step_exp_table);
		roundoff_differences.push_back(range);
		printf(
			"TRACE_INTERVAL stage=roundoff_polynomial_difference state=%u "
			"lower=%.17g upper=%.17g\n",
			i, range.inf(), range.sup());
	}

	bool candidate_contains = true;
	for(unsigned int i = 0; i < range_dim; ++i)
	{
		picard_image.tms[i].remainder += roundoff_differences[i];
		if(!picard_image.tms[i].remainder.subseteq(picard.tms[i].remainder))
		{
			candidate_contains = false;
		}
	}
	printf("TRACE_CHECK name=candidate_contains_picard_image value=%d\n", candidate_contains ? 1 : 0);
	print_interval_tmv("validated_picard_image", picard_image, accepted.domain);
	print_point_box(
		"validated_picard_image", "endpoint_lower_corner", picard_image,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"validated_picard_image", "endpoint_upper_corner", picard_image,
		accepted.domain, config.h, 1.0);
	if(!candidate_contains)
	{
		return 5;
	}

	for(unsigned int i = 0; i < range_dim; ++i)
	{
		picard.tms[i].remainder = picard_image.tms[i].remainder;
	}

	bool finished = false;
	for(int refinement = 0;
		!finished && refinement <= MAX_REFINEMENT_STEPS; ++refinement)
	{
		finished = true;
		vector<Interval> new_remainders;
		picard.Picard_ctrunc_normal_remainder(
			new_remainders, ode.expressions, setting.tm_setting.step_exp_table[1],
			setting.tm_setting.order, intermediate_ranges, setting.g_setting);
		for(unsigned int i = 0; i < range_dim; ++i)
		{
			printf(
				"TRACE_INTERVAL stage=refinement_%d_raw state=%u lower=%.17g upper=%.17g\n",
				refinement, i, new_remainders[i].inf(), new_remainders[i].sup());
			new_remainders[i] += roundoff_differences[i];
			const bool subset = new_remainders[i].subseteq(picard.tms[i].remainder);
			printf(
				"TRACE_INTERVAL stage=refinement_%d_with_roundoff state=%u "
				"lower=%.17g upper=%.17g\n",
				refinement, i, new_remainders[i].inf(), new_remainders[i].sup());
			printf(
				"TRACE_CHECK name=refinement_%d_state_%u_subset value=%d\n",
				refinement, i, subset ? 1 : 0);
			if(subset)
			{
				if(picard.tms[i].remainder.widthRatio(new_remainders[i]) <= STOP_RATIO)
				{
					finished = false;
				}
				picard.tms[i].remainder = new_remainders[i];
			}
			else
			{
				finished = true;
				break;
			}
		}
		const string refinement_stage =
			"refinement_" + to_string(refinement) + "_accepted_tmv";
		print_real_tmv(refinement_stage.c_str(), picard, accepted.domain);
		print_point_box(
			refinement_stage.c_str(), "endpoint_lower_corner", picard,
			accepted.domain, config.h, -1.0);
		print_point_box(
			refinement_stage.c_str(), "endpoint_upper_corner", picard,
			accepted.domain, config.h, 1.0);
	}

	print_real_tmv("accepted_mirror_tmv_pre", picard, accepted.domain);
	print_point_box(
		"accepted_mirror_tmv_pre", "endpoint_lower_corner", picard,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"accepted_mirror_tmv_pre", "endpoint_upper_corner", picard,
		accepted.domain, config.h, 1.0);
	print_real_tmv("accepted_native_tmv_pre", accepted.tmvPre, accepted.domain);
	print_point_box(
		"accepted_native_tmv_pre", "endpoint_lower_corner", accepted.tmvPre,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"accepted_native_tmv_pre", "endpoint_upper_corner", accepted.tmvPre,
		accepted.domain, config.h, 1.0);
	print_real_tmv("accepted_native_tmv_right", accepted.tmv, accepted.domain);

	TaylorModelVec<Real> composed;
	accepted.compose(
		composed, config.order, setting.tm_setting.cutoff_threshold);
	print_real_tmv("composed_flowpipe", composed, accepted.domain);
	print_point_box(
		"composed_flowpipe", "endpoint_lower_corner", composed,
		accepted.domain, config.h, -1.0);
	print_point_box(
		"composed_flowpipe", "endpoint_upper_corner", composed,
		accepted.domain, config.h, 1.0);

	vector<Interval> endpoint_domain = accepted.domain;
	endpoint_domain[0] = Interval(config.h);
	vector<Interval> endpoint_raw;
	composed.intEval(endpoint_raw, endpoint_domain);
	for(unsigned int i = 0; i < endpoint_raw.size(); ++i)
	{
		printf(
			"TRACE_BOX stage=endpoint_raw scope=full_initial_interval state=%u "
			"lower=%.17g upper=%.17g\n",
			i, endpoint_raw[i].inf(), endpoint_raw[i].sup());
	}

	TaylorModelVec<Real> endpoint_collapsed;
	composed.evaluate_time(
		endpoint_collapsed, endpoint_powers(accepted.domain[0], config.order));
	vector<Interval> collapsed_domain = accepted.domain;
	collapsed_domain[0] = Interval(0);
	print_real_tmv("endpoint_collapsed", endpoint_collapsed, collapsed_domain);

	printf("TRACE_UNAVAILABLE stage=endpoint_tightened reason=no_distinct_stock_field\n");
	printf("TRACE_UNAVAILABLE stage=repaired_hull reason=prohibited_not_computed\n");
	return 0;
}
