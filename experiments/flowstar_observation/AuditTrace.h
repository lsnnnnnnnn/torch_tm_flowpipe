#ifndef FLOWSTAR_AUDIT_TRACE_H_
#define FLOWSTAR_AUDIT_TRACE_H_

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace flowstar_audit
{

struct Context
{
	int accepted_step_index;
	int retry_index;
	long attempt_index;
	double t_pre;
	double h_attempt;
	Context() : accepted_step_index(-1), retry_index(-1), attempt_index(-1), t_pre(0), h_attempt(0) {}
};

inline Context & context()
{
	static Context value;
	return value;
}

inline bool enabled()
{
	const char *path = std::getenv("FLOWSTAR_AUDIT_TRACE");
	return path != NULL && path[0] != '\0';
}

inline std::string escaped(const std::string &value)
{
	std::ostringstream out;
	for(std::string::const_iterator it = value.begin(); it != value.end(); ++it)
	{
		switch(*it)
		{
		case '\\': out << "\\\\"; break;
		case '"': out << "\\\""; break;
		case '\n': out << "\\n"; break;
		case '\r': out << "\\r"; break;
		case '\t': out << "\\t"; break;
		default: out << *it;
		}
	}
	return out.str();
}

inline std::string quoted(const std::string &value)
{
	return std::string("\"") + escaped(value) + "\"";
}

inline std::string number(const double value)
{
	std::ostringstream decimal, hex;
	decimal << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
	hex << std::hexfloat << value;
	return std::string("{\"decimal\":") + quoted(decimal.str()) + ",\"hex\":" + quoted(hex.str()) + "}";
}

inline std::string interval(const flowstar::Interval &value)
{
	return std::string("{\"lower\":") + number(value.inf()) + ",\"upper\":" + number(value.sup()) + "}";
}

inline std::string nullable_interval(const flowstar::Interval *value)
{
	return value == NULL ? "null" : interval(*value);
}

inline std::string base(const std::string &record_type, const std::string &stage, const int component)
{
	const char *run = std::getenv("FLOWSTAR_AUDIT_RUN_ID");
	const char *commit = std::getenv("FLOWSTAR_AUDIT_SOURCE_COMMIT");
	Context &ctx = context();
	std::ostringstream out;
	out << "{\"record_type\":" << quoted(record_type)
		<< ",\"tool\":\"flowstar\""
		<< ",\"source_commit\":" << quoted(commit == NULL ? "unknown" : commit)
		<< ",\"run_id\":" << quoted(run == NULL ? "flowstar-observation" : run)
		<< ",\"accepted_step_index\":" << ctx.accepted_step_index
		<< ",\"attempt_index\":" << ctx.attempt_index
		<< ",\"retry_index\":" << ctx.retry_index
		<< ",\"t_pre\":" << number(ctx.t_pre)
		<< ",\"h_attempt\":" << number(ctx.h_attempt)
		<< ",\"state_component\":" << component
		<< ",\"stage\":" << quoted(stage);
	return out.str();
}

inline void write(const std::string &line)
{
	if(!enabled()) return;
	std::ofstream out(std::getenv("FLOWSTAR_AUDIT_TRACE"), std::ios::out | std::ios::app);
	out << line << "\n";
}

inline void set_step_context(const int accepted_step_index, const double t_pre)
{
	if(!enabled()) return;
	context().accepted_step_index = accepted_step_index;
	context().retry_index = -1;
	context().t_pre = t_pre;
}

inline void begin_attempt(const double h_attempt)
{
	if(!enabled()) return;
	++context().retry_index;
	++context().attempt_index;
	context().h_attempt = h_attempt;
}

inline void emit_interval_stage(
	const std::string &stage,
	const int component,
	const flowstar::Interval *polynomial_range,
	const flowstar::Interval *remainder,
	const bool accepted,
	const std::string &reason,
	const int support_size,
	const double *center,
	const double *scale)
{
	if(!enabled()) return;
	std::ostringstream out;
	out << base("transition", stage, component)
		<< ",\"accepted\":" << (accepted ? "true" : "false")
		<< ",\"rejection_reason\":" << quoted(reason)
		<< ",\"basis_variable_order\":[\"tau\",\"r0\",\"r1\",\"r2\"]"
		<< ",\"center\":" << (center == NULL ? "null" : number(*center))
		<< ",\"normalization_scale\":" << (scale == NULL ? "null" : number(*scale))
		<< ",\"support_size\":" << support_size
		<< ",\"polynomial_range\":" << nullable_interval(polynomial_range)
		<< ",\"remainder\":" << nullable_interval(remainder)
		<< ",\"cutoff_discarded_terms\":null"
		<< ",\"truncation_discarded_terms\":null"
		<< ",\"insertion_truncation_remainder\":null"
		<< ",\"right_map_remainder\":null"
		<< ",\"integration_overflow\":null"
		<< ",\"self_map_candidate_box\":null"
		<< ",\"self_map_image\":null"
		<< ",\"violation_margin\":null}";
	write(out.str());
	if(remainder != NULL)
	{
		write(base("remainder", stage, component) + ",\"accepted\":" + (accepted ? "true" : "false")
			+ ",\"rejection_reason\":" + quoted(reason) + ",\"interval\":" + interval(*remainder) + "}");
	}
}

inline void coefficient_bounds(const flowstar::Real &value, double &lo, double &hi)
{
	lo = value.getValue_RNDD();
	hi = value.getValue_RNDU();
}

inline void coefficient_bounds(const flowstar::Interval &value, double &lo, double &hi)
{
	lo = value.inf();
	hi = value.sup();
}

template <class DATA_TYPE>
inline void emit_tmv(
	const std::string &stage,
	const flowstar::TaylorModelVec<DATA_TYPE> &tmv,
	const std::vector<flowstar::Interval> &step_exp_table,
	const bool accepted,
	const std::string &reason,
	const std::vector<double> *centers,
	const std::vector<double> *scales)
{
	if(!enabled()) return;
	for(unsigned int component = 0; component < tmv.tms.size(); ++component)
	{
		flowstar::Interval poly_range;
		tmv.tms[component].expansion.intEvalNormal(poly_range, step_exp_table);
		double center_value = 0, scale_value = 0;
		const double *center = NULL, *scale = NULL;
		if(centers != NULL && component < centers->size())
		{
			center_value = (*centers)[component];
			center = &center_value;
		}
		if(scales != NULL && component < scales->size())
		{
			scale_value = (*scales)[component];
			scale = &scale_value;
		}
		emit_interval_stage(stage, component, &poly_range, &tmv.tms[component].remainder, accepted, reason,
			static_cast<int>(tmv.tms[component].expansion.terms.size()), center, scale);
		int term_index = 0;
		for(typename std::list<flowstar::Term<DATA_TYPE> >::const_iterator term = tmv.tms[component].expansion.terms.begin();
			term != tmv.tms[component].expansion.terms.end(); ++term, ++term_index)
		{
			double coefficient_lo = 0, coefficient_hi = 0;
			coefficient_bounds(term->auditCoefficient(), coefficient_lo, coefficient_hi);
			std::ostringstream degrees;
			degrees << "[";
			const std::vector<unsigned int> &items = term->auditDegrees();
			for(unsigned int i = 0; i < items.size(); ++i)
			{
				if(i > 0) degrees << ",";
				degrees << items[i];
			}
			degrees << "]";
			std::ostringstream out;
			out << base("polynomial_term", stage, component)
				<< ",\"accepted\":" << (accepted ? "true" : "false")
				<< ",\"rejection_reason\":" << quoted(reason)
				<< ",\"term_index\":" << term_index
				<< ",\"exponent_tuple\":" << degrees.str()
				<< ",\"degree\":" << term->degree()
				<< ",\"coefficient\":{\"lower\":" << number(coefficient_lo) << ",\"upper\":" << number(coefficient_hi) << "}}";
			write(out.str());
		}
	}
}

inline void emit_acceptance(
	const std::vector<flowstar::Interval> &images,
	const std::vector<flowstar::Interval> &targets,
	const bool accepted,
	const std::string &reason)
{
	if(!enabled()) return;
	for(unsigned int component = 0; component < images.size(); ++component)
	{
		const double lower_margin = images[component].inf() - targets[component].inf();
		const double upper_margin = targets[component].sup() - images[component].sup();
		const double margin = lower_margin < upper_margin ? lower_margin : upper_margin;
		std::ostringstream out;
		out << base("transition", "acceptance_predicate", component)
			<< ",\"accepted\":" << (accepted ? "true" : "false")
			<< ",\"rejection_reason\":" << quoted(reason)
			<< ",\"basis_variable_order\":[\"tau\",\"r0\",\"r1\",\"r2\"]"
			<< ",\"center\":null,\"normalization_scale\":null,\"support_size\":null,\"polynomial_range\":null"
			<< ",\"remainder\":" << interval(images[component])
			<< ",\"cutoff_discarded_terms\":null,\"truncation_discarded_terms\":null"
			<< ",\"insertion_truncation_remainder\":null,\"right_map_remainder\":null,\"integration_overflow\":null"
			<< ",\"self_map_candidate_box\":" << interval(targets[component])
			<< ",\"self_map_image\":" << interval(images[component])
			<< ",\"violation_margin\":" << number(margin) << "}";
		write(out.str());
	}
}

inline void emit_scheduler(const bool accepted, const std::string &reason)
{
	if(!enabled()) return;
	write(base("acceptance_attempt", "scheduler", -1)
		+ ",\"accepted\":" + (accepted ? "true" : "false")
		+ ",\"rejection_reason\":" + quoted(reason) + "}");
}

inline void emit_missing_stage(const std::string &stage, const bool accepted, const std::string &reason)
{
	if(!enabled()) return;
	write(base("transition", stage, -1)
		+ ",\"accepted\":" + (accepted ? "true" : "false")
		+ ",\"rejection_reason\":" + quoted(reason)
		+ ",\"basis_variable_order\":[\"tau\",\"r0\",\"r1\",\"r2\"]"
		+ ",\"center\":null,\"normalization_scale\":null,\"support_size\":null"
		+ ",\"polynomial_range\":null,\"remainder\":null"
		+ ",\"cutoff_discarded_terms\":null,\"truncation_discarded_terms\":null"
		+ ",\"insertion_truncation_remainder\":null,\"right_map_remainder\":null"
		+ ",\"integration_overflow\":null,\"self_map_candidate_box\":null"
		+ ",\"self_map_image\":null,\"violation_margin\":null}");
}

template <class FLOWPIPE>
inline void emit_observed_step(
	const FLOWPIPE &current,
	const FLOWPIPE &result,
	const std::vector<flowstar::Interval> &step_exp_table,
	const bool accepted)
{
	if(!enabled()) return;
	begin_attempt(step_exp_table.size() > 1 ? step_exp_table[1].sup() : 0.0);
	const std::string reason = accepted ? "" : "stock advance returned failure";
	emit_missing_stage("step_pre_state", accepted, "the physical composed pre-state is not exported at the stock scheduler observation point");
	emit_missing_stage("insertion_input", accepted, "the pre-scaling insert_ctrunc result is local to advance_adaptive_stepsize and is not exported at this hook");
	emit_missing_stage("insertion_output", accepted, "the pre-scaling insertion result is not exported; result.tmv is recorded only under its stored right-map identity");
	emit_tmv("right_map_input", current.tmv, step_exp_table, accepted, reason, NULL, NULL);
	emit_tmv("right_map_output", result.tmv, step_exp_table, accepted, reason, NULL, NULL);
	emit_missing_stage("normalized_reset_input", accepted, "stock Flowstar stores a left/right composition rather than the Torch normalized-reset object");
	emit_missing_stage("normalized_reset_output", accepted, "stock Flowstar stores a left/right composition rather than the Torch normalized-reset object");
	emit_tmv("raw_picard_image", result.tmvPre, step_exp_table, accepted, reason, NULL, NULL);
	emit_missing_stage("truncation_cutoff", accepted, "not exported at scheduler observation point; see diagnostic probe lane");
	emit_missing_stage("acceptance_predicate", accepted, accepted ? "" : reason);
	emit_tmv("next_step_pre_state", result.tmvPre, step_exp_table, accepted, reason, NULL, NULL);
	emit_scheduler(accepted, reason);
}

} // namespace flowstar_audit

#endif
