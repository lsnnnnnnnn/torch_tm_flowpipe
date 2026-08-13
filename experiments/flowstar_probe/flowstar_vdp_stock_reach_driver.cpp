#include "Continuous.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

#ifdef FLOWSTAR_CAUSAL_OBSERVE
ofstream observer_output;

string canonical_real(const Real &value)
{
    const string result = value.auditCanonicalBinary();
    if (result.empty())
    {
        throw runtime_error("non-finite MPFR value in observation hook");
    }
    return result;
}

string canonical_interval(const Interval &value)
{
    Real lower;
    Real upper;
    value.inf(lower);
    value.sup(upper);
    return canonical_real(lower) + ".." + canonical_real(upper);
}

string canonical_tmv(const TaylorModelVec<Real> &tmv)
{
    ostringstream output;
    output << "components=" << tmv.tms.size();
    for (size_t component = 0; component < tmv.tms.size(); ++component)
    {
        const TaylorModel<Real> &model = tmv.tms[component];
        output << "|c=" << component << ";terms=" << model.expansion.terms.size();
        size_t term_index = 0;
        for (list<Term<Real> >::const_iterator iterator = model.expansion.terms.begin();
             iterator != model.expansion.terms.end(); ++iterator, ++term_index)
        {
            output << ";t=" << term_index << ";e=";
            const vector<unsigned int> &degrees = iterator->auditDegrees();
            for (size_t index = 0; index < degrees.size(); ++index)
            {
                if (index > 0)
                {
                    output << '.';
                }
                output << degrees[index];
            }
            output << ";a=" << canonical_real(iterator->auditCoefficient());
        }
        output << ";r=" << canonical_interval(model.remainder);
    }
    return output.str();
}

string canonical_queue(const Symbolic_Remainder &queue)
{
    ostringstream output;
    output << "max=" << queue.max_size << "|scalars=" << queue.scalars.size();
    for (size_t index = 0; index < queue.scalars.size(); ++index)
    {
        output << ";s=" << index << ':' << canonical_real(queue.scalars[index]);
    }
    output << "|J=" << queue.J.size();
    for (size_t matrix_index = 0; matrix_index < queue.J.size(); ++matrix_index)
    {
        const Matrix<Interval> &matrix = queue.J[matrix_index];
        output << ";jm=" << matrix_index << ':' << matrix.rows() << 'x' << matrix.cols();
        for (int row = 0; row < matrix.rows(); ++row)
        {
            for (int column = 0; column < matrix.cols(); ++column)
            {
                output << ':' << canonical_interval(matrix[row][column]);
            }
        }
    }
    output << "|Phi=" << queue.Phi_L.size();
    for (size_t matrix_index = 0; matrix_index < queue.Phi_L.size(); ++matrix_index)
    {
        const Matrix<Real> &matrix = queue.Phi_L[matrix_index];
        output << ";pm=" << matrix_index << ':' << matrix.rows() << 'x' << matrix.cols();
        for (int row = 0; row < matrix.rows(); ++row)
        {
            for (int column = 0; column < matrix.cols(); ++column)
            {
                output << ':' << canonical_real(matrix[row][column]);
            }
        }
    }
    return output.str();
}

string canonical_domain(const vector<Interval> &domain)
{
    ostringstream output;
    output << domain.size();
    for (size_t index = 0; index < domain.size(); ++index)
    {
        output << '|' << index << ':' << canonical_interval(domain[index]);
    }
    return output.str();
}
#endif

struct Bounds
{
    double lo;
    double hi;
};

string decimal17(const double value)
{
    ostringstream stream;
    stream << setprecision(17) << value;
    return stream.str();
}

string hexfloat_text(const double value)
{
    ostringstream stream;
    stream << hexfloat << value;
    return stream.str();
}

Bounds outward_bounds(const Interval &interval)
{
    Bounds result;
    result.lo = interval.inf();
    result.hi = interval.sup();
    if (!isfinite(result.lo) || !isfinite(result.hi) || result.lo > result.hi)
    {
        throw runtime_error("invalid outward interval");
    }
    return result;
}

void write_bound_fields(ofstream &output, const Bounds &bounds)
{
    output << ',' << decimal17(bounds.lo)
           << ',' << decimal17(bounds.hi)
           << ',' << hexfloat_text(bounds.lo)
           << ',' << hexfloat_text(bounds.hi);
}

void hull_assign(Bounds &target, const Bounds &value, const bool initialized)
{
    if (!initialized)
    {
        target = value;
        return;
    }
    if (value.lo < target.lo)
    {
        target.lo = value.lo;
    }
    if (value.hi > target.hi)
    {
        target.hi = value.hi;
    }
}

vector<Interval> endpoint_step_table(const Taylor_Model_Setting &setting)
{
    vector<Interval> result;
    for (size_t index = 0; index < setting.step_end_exp_table.size(); ++index)
    {
        result.push_back(Interval(setting.step_end_exp_table[index]));
    }
    return result;
}

size_t term_count(const TaylorModelVec<Real> &tmv)
{
    size_t result = 0;
    for (size_t component = 0; component < tmv.tms.size(); ++component)
    {
        result += tmv.tms[component].expansion.terms.size();
    }
    return result;
}

void write_header(ofstream &output)
{
    output
        << "schema,mode,step,t_after,t_after_hex,h,h_hex,status_code,"
        << "endpoint_x_lo,endpoint_x_hi,endpoint_x_lo_hex,endpoint_x_hi_hex,"
        << "endpoint_y_lo,endpoint_y_hi,endpoint_y_lo_hex,endpoint_y_hi_hex,"
        << "segment_x_lo,segment_x_hi,segment_x_lo_hex,segment_x_hi_hex,"
        << "segment_y_lo,segment_y_hi,segment_y_lo_hex,segment_y_hi_hex,"
        << "prefix_x_lo,prefix_x_hi,prefix_x_lo_hex,prefix_x_hi_hex,"
        << "prefix_y_lo,prefix_y_hi,prefix_y_lo_hex,prefix_y_hi_hex,"
        << "tmvpre_term_count,tmv_term_count,tmvpre_component_count,tmv_component_count,"
        << "tmvpre_remainder_x_lo,tmvpre_remainder_x_hi,tmvpre_remainder_x_lo_hex,tmvpre_remainder_x_hi_hex,"
        << "tmvpre_remainder_y_lo,tmvpre_remainder_y_hi,tmvpre_remainder_y_lo_hex,tmvpre_remainder_y_hi_hex,"
        << "domain_time_lo,domain_time_hi,domain_time_lo_hex,domain_time_hi_hex,"
        << "queue_J_post_return,queue_Phi_L_post_return,queue_scalars,queue_max_size\n";
}

void write_flowpipe_row(
    ofstream &output,
    const string &mode,
    const size_t step,
    const int status,
    const double fixed_step,
    const Flowpipe &flowpipe,
    const Taylor_Model_Setting &setting,
    const Symbolic_Remainder &symbolic_remainder,
    Bounds prefix[2],
    bool &prefix_initialized)
{
    vector<Interval> segment;
    flowpipe.intEvalNormal(
        segment,
        setting.step_exp_table,
        setting.order,
        setting.cutoff_threshold);
    const vector<Interval> endpoint_table = endpoint_step_table(setting);
    vector<Interval> endpoint;
    flowpipe.intEvalNormal(
        endpoint,
        endpoint_table,
        setting.order,
        setting.cutoff_threshold);
    if (segment.size() < 2 || endpoint.size() < 2)
    {
        throw runtime_error("stock Flow* returned fewer than two state dimensions");
    }
    const Bounds endpoint_x = outward_bounds(endpoint[0]);
    const Bounds endpoint_y = outward_bounds(endpoint[1]);
    const Bounds segment_x = outward_bounds(segment[0]);
    const Bounds segment_y = outward_bounds(segment[1]);
    hull_assign(prefix[0], segment_x, prefix_initialized);
    hull_assign(prefix[1], segment_y, prefix_initialized);
    prefix_initialized = true;

    output << "flowstar_vdp_stock_reach_v1"
           << ',' << mode
           << ',' << step
           << ',' << decimal17(static_cast<double>(step) * fixed_step)
           << ',' << hexfloat_text(static_cast<double>(step) * fixed_step)
           << ',' << decimal17(fixed_step)
           << ',' << hexfloat_text(fixed_step)
           << ',' << status;
    write_bound_fields(output, endpoint_x);
    write_bound_fields(output, endpoint_y);
    write_bound_fields(output, segment_x);
    write_bound_fields(output, segment_y);
    write_bound_fields(output, prefix[0]);
    write_bound_fields(output, prefix[1]);
    output << ',' << term_count(flowpipe.tmvPre)
           << ',' << term_count(flowpipe.tmv)
           << ',' << flowpipe.tmvPre.tms.size()
           << ',' << flowpipe.tmv.tms.size();
    if (flowpipe.tmvPre.tms.size() < 2)
    {
        throw runtime_error("stock Flow* tmvPre has fewer than two state dimensions");
    }
    write_bound_fields(output, outward_bounds(flowpipe.tmvPre.tms[0].remainder));
    write_bound_fields(output, outward_bounds(flowpipe.tmvPre.tms[1].remainder));
    if (flowpipe.domain.empty())
    {
        throw runtime_error("stock Flow* returned an empty domain");
    }
    write_bound_fields(output, outward_bounds(flowpipe.domain[0]));
    output << ',' << symbolic_remainder.J.size()
           << ',' << symbolic_remainder.Phi_L.size()
           << ',' << symbolic_remainder.scalars.size()
           << ',' << symbolic_remainder.max_size
           << '\n';
}

void configure(
    Variables &variables,
    ODE<Real> *&ode,
    Computational_Setting *&setting,
    Flowpipe *&initial,
    const double fixed_step,
    const unsigned int order)
{
    const int x_id = variables.declareVar("x");
    const int y_id = variables.declareVar("y");
    variables.declareVar("t");
    ode = new ODE<Real>({"y", "(1 - x^2) * y - x", "1"}, variables);
    setting = new Computational_Setting(variables);
    setting->printOff();
    if (!setting->setFixedStepsize(fixed_step, order))
    {
        throw runtime_error("Flow* rejected fixed-step configuration");
    }
    setting->setCutoffThreshold(1e-10);
    vector<Interval> target(ode->expressions.size(), Interval(-1e-4, 1e-4));
    setting->setRemainderEstimation(target);
    vector<Interval> box(variables.size());
    box[x_id] = Interval(1.1, 1.4);
    box[y_id] = Interval(2.35, 2.45);
    initial = new Flowpipe(box);
}

int run_oneshot(
    ofstream &output,
    ODE<Real> &ode,
    Computational_Setting &setting,
    const Flowpipe &initial,
    const double horizon,
    const double fixed_step,
    const unsigned int max_queue,
    size_t &accepted)
{
    Result_of_Reachability result;
    Symbolic_Remainder symbolic_remainder(initial, max_queue);
    vector<Constraint> safe_set;
    ode.reach(result, initial, horizon, setting, safe_set, symbolic_remainder);
    Bounds prefix[2];
    bool prefix_initialized = false;
    size_t step = 0;
    for (list<Flowpipe>::const_iterator iterator = result.flowpipes.begin();
         iterator != result.flowpipes.end(); ++iterator)
    {
        ++step;
        write_flowpipe_row(
            output,
            "oneshot",
            step,
            result.status,
            fixed_step,
            *iterator,
            setting.tm_setting,
            symbolic_remainder,
            prefix,
            prefix_initialized);
    }
    accepted = step;
    return result.status;
}

int run_stepwise(
    ofstream &output,
    ODE<Real> &ode,
    Computational_Setting &setting,
    const Flowpipe &initial,
    const size_t requested_steps,
    const double fixed_step,
    const unsigned int max_queue,
    size_t &accepted)
{
    Symbolic_Remainder symbolic_remainder(initial, max_queue);
    vector<Constraint> safe_set;
    Flowpipe current(initial);
    Bounds prefix[2];
    bool prefix_initialized = false;
    int final_status = 0;
    accepted = 0;
    for (size_t step = 1; step <= requested_steps; ++step)
    {
        Result_of_Reachability result;
        ode.reach(result, current, fixed_step, setting, safe_set, symbolic_remainder);
        final_status = result.status;
        if (result.flowpipes.size() != 1)
        {
            break;
        }
        current = result.flowpipes.back();
        write_flowpipe_row(
            output,
            "stepwise",
            step,
            result.status,
            fixed_step,
            current,
            setting.tm_setting,
            symbolic_remainder,
            prefix,
            prefix_initialized);
        accepted = step;
        if (!result.isCompleted())
        {
            break;
        }
    }
    return final_status;
}

} // namespace

#ifdef FLOWSTAR_CAUSAL_OBSERVE
void flowstar::flowstar_causal_observe_reach_step(
    const Flowpipe &flowpipe,
    const Symbolic_Remainder &symbolic_remainder,
    const unsigned long accepted_step,
    const double t_before,
    const double step,
    const char *phase)
{
    if (!observer_output)
    {
        throw runtime_error("observation hook output is not open");
    }
    observer_output
        << "flowstar_actual_path_observation_v1"
        << ',' << phase
        << ',' << accepted_step
        << ',' << decimal17(t_before)
        << ',' << hexfloat_text(t_before)
        << ',' << decimal17(step)
        << ',' << hexfloat_text(step)
        << ',' << symbolic_remainder.J.size()
        << ',' << symbolic_remainder.Phi_L.size()
        << ',' << canonical_tmv(flowpipe.tmvPre)
        << ',' << canonical_tmv(flowpipe.tmv)
        << ',' << canonical_domain(flowpipe.domain)
        << ',' << canonical_queue(symbolic_remainder)
        << '\n';
}
#endif

int main(int argc, char **argv)
{
    const int expected_arguments =
#ifdef FLOWSTAR_CAUSAL_OBSERVE
        7;
#else
        6;
#endif
    if (argc != expected_arguments)
    {
        cerr << "usage: flowstar_vdp_stock_reach_driver OUTPUT_CSV SUMMARY_JSON "
             << "oneshot|stepwise MAX_QUEUE REQUESTED_STEPS"
#ifdef FLOWSTAR_CAUSAL_OBSERVE
             << " OBSERVER_CSV"
#endif
             << "\n";
        return 2;
    }
    const string csv_path = argv[1];
    const string summary_path = argv[2];
    const string mode = argv[3];
    const unsigned int max_queue = static_cast<unsigned int>(strtoul(argv[4], NULL, 10));
    const size_t requested_steps = static_cast<size_t>(strtoull(argv[5], NULL, 10));
    if ((mode != "oneshot" && mode != "stepwise") || max_queue == 0 || requested_steps == 0)
    {
        cerr << "invalid driver arguments\n";
        return 2;
    }

    const double fixed_step = 0.01;
    const unsigned int order = 4;
    Variables variables;
    ODE<Real> *ode = NULL;
    Computational_Setting *setting = NULL;
    Flowpipe *initial = NULL;
    try
    {
#ifdef FLOWSTAR_CAUSAL_OBSERVE
        observer_output.open(argv[6]);
        if (!observer_output)
        {
            throw runtime_error("cannot open observer CSV");
        }
        observer_output
            << "schema,phase,step,t_before,t_before_hex,h,h_hex,queue_J_size,"
            << "queue_Phi_L_size,tmvPre_canonical,tmv_canonical,domain_canonical,queue_canonical\n";
#endif
        configure(variables, ode, setting, initial, fixed_step, order);
        ofstream output(csv_path.c_str());
        if (!output)
        {
            throw runtime_error("cannot open output CSV");
        }
        write_header(output);
        size_t accepted = 0;
        int status = 0;
        if (mode == "oneshot")
        {
            status = run_oneshot(
                output,
                *ode,
                *setting,
                *initial,
                static_cast<double>(requested_steps) * fixed_step,
                fixed_step,
                max_queue,
                accepted);
        }
        else
        {
            status = run_stepwise(
                output,
                *ode,
                *setting,
                *initial,
                requested_steps,
                fixed_step,
                max_queue,
                accepted);
        }
        output.close();
        ofstream summary(summary_path.c_str());
        if (!summary)
        {
            throw runtime_error("cannot open summary JSON");
        }
        summary << "{\n"
                << "  \"schema\": \"flowstar_vdp_stock_reach_summary_v1\",\n"
                << "  \"actual_public_entry\": \"ODE<Real>::reach(Result_of_Reachability&, ..., Symbolic_Remainder&)\",\n"
                << "  \"copies_flowpipe_advance\": false,\n"
                << "  \"mode\": \"" << mode << "\",\n"
                << "  \"fixed_step\": \"0.01\",\n"
                << "  \"fixed_step_hex\": \"" << hexfloat_text(fixed_step) << "\",\n"
                << "  \"order\": " << order << ",\n"
                << "  \"queue_max_size\": " << max_queue << ",\n"
                << "  \"requested_steps\": " << requested_steps << ",\n"
                << "  \"accepted_steps\": " << accepted << ",\n"
                << "  \"result_status_code\": " << status << "\n"
                << "}\n";
        delete initial;
        delete setting;
        delete ode;
#ifdef FLOWSTAR_CAUSAL_OBSERVE
        observer_output.close();
#endif
        return accepted == requested_steps ? 0 : 1;
    }
    catch (const exception &error)
    {
        delete initial;
        delete setting;
        delete ode;
        cerr << error.what() << '\n';
        return 3;
    }
}
