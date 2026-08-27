#include "Continuous.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

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
    Bounds result = {interval.inf(), interval.sup()};
    if (!isfinite(result.lo) || !isfinite(result.hi) || result.lo > result.hi)
    {
        throw runtime_error("invalid outward interval");
    }
    return result;
}

void write_bounds(ofstream &output, const Bounds &value)
{
    output << ',' << decimal17(value.lo)
           << ',' << decimal17(value.hi)
           << ',' << decimal17(value.hi - value.lo)
           << ',' << hexfloat_text(value.lo)
           << ',' << hexfloat_text(value.hi);
}

void hull(Bounds &target, const Bounds &value)
{
    target.lo = min(target.lo, value.lo);
    target.hi = max(target.hi, value.hi);
}

vector<Interval> endpoint_table(const Taylor_Model_Setting &setting)
{
    vector<Interval> result;
    for (size_t index = 0; index < setting.step_end_exp_table.size(); ++index)
    {
        result.push_back(Interval(setting.step_end_exp_table[index]));
    }
    return result;
}

void write_header(ofstream &output)
{
    output
        << "schema,step,t_after,t_after_hex,h,h_hex,"
        << "endpoint_x_lo,endpoint_x_hi,endpoint_x_width,endpoint_x_lo_hex,endpoint_x_hi_hex,"
        << "endpoint_y_lo,endpoint_y_hi,endpoint_y_width,endpoint_y_lo_hex,endpoint_y_hi_hex,"
        << "tube_x_lo,tube_x_hi,tube_x_width,tube_x_lo_hex,tube_x_hi_hex,"
        << "tube_y_lo,tube_y_hi,tube_y_width,tube_y_lo_hex,tube_y_hi_hex,"
        << "prefix_x_lo,prefix_x_hi,prefix_x_width,prefix_x_lo_hex,prefix_x_hi_hex,"
        << "prefix_y_lo,prefix_y_hi,prefix_y_width,prefix_y_lo_hex,prefix_y_hi_hex\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "usage: flowstar_brusselator_second_system OUTPUT_CSV SUMMARY_JSON\n";
        return 2;
    }
    const double fixed_step = 0.02;
    const unsigned int order = 6;
    const size_t requested_steps = 1000;
    const double horizon = 20.0;
    const unsigned int queue_capacity = 1000;

    try
    {
        Variables variables;
        const int x_id = variables.declareVar("x");
        const int y_id = variables.declareVar("y");
        variables.declareVar("t");
        ODE<Real> ode({"1 + x*(x*y - 4)", "x*(3 - x*y)", "1"}, variables);
        Computational_Setting setting(variables);
        setting.printOff();
        if (!setting.setFixedStepsize(fixed_step, order))
        {
            throw runtime_error("Flow* rejected the frozen fixed-step request");
        }
        setting.setCutoffThreshold(1e-10);
        vector<Interval> remainder(variables.size(), Interval(-1e-4, 1e-4));
        setting.setRemainderEstimation(remainder);
        vector<Interval> box(variables.size());
        box[x_id] = Interval("1.48", "1.52");
        box[y_id] = Interval("2.98", "3.02");
        Flowpipe initial(box);
        Symbolic_Remainder symbolic_remainder(initial, queue_capacity);
        vector<Constraint> safe_set;
        Result_of_Reachability result;

        const chrono::steady_clock::time_point started = chrono::steady_clock::now();
        const clock_t core_started = clock();
        ode.reach(result, initial, horizon, setting, safe_set, symbolic_remainder);
        const clock_t core_finished = clock();
        const double wall_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - started).count();
        const double core_seconds = static_cast<double>(core_finished - core_started) / CLOCKS_PER_SEC;

        ofstream output(argv[1]);
        if (!output)
        {
            throw runtime_error("cannot open Flow* CSV output");
        }
        write_header(output);
        Bounds prefix_x = {1.48, 1.52};
        Bounds prefix_y = {2.98, 3.02};
        const vector<Interval> endpoint = endpoint_table(setting.tm_setting);
        size_t step = 0;
        for (list<Flowpipe>::const_iterator iterator = result.flowpipes.begin();
             iterator != result.flowpipes.end(); ++iterator)
        {
            ++step;
            vector<Interval> endpoint_box;
            vector<Interval> tube_box;
            iterator->intEvalNormal(
                endpoint_box,
                endpoint,
                setting.tm_setting.order,
                setting.tm_setting.cutoff_threshold);
            iterator->intEvalNormal(
                tube_box,
                setting.tm_setting.step_exp_table,
                setting.tm_setting.order,
                setting.tm_setting.cutoff_threshold);
            if (endpoint_box.size() < 2 || tube_box.size() < 2)
            {
                throw runtime_error("Flow* returned fewer than two plant dimensions");
            }
            const Bounds endpoint_x = outward_bounds(endpoint_box[0]);
            const Bounds endpoint_y = outward_bounds(endpoint_box[1]);
            const Bounds tube_x = outward_bounds(tube_box[0]);
            const Bounds tube_y = outward_bounds(tube_box[1]);
            hull(prefix_x, tube_x);
            hull(prefix_y, tube_y);
            output << "flowstar_brusselator_second_system_v1"
                   << ',' << step
                   << ',' << decimal17(static_cast<double>(step) * fixed_step)
                   << ',' << hexfloat_text(static_cast<double>(step) * fixed_step)
                   << ',' << decimal17(fixed_step)
                   << ',' << hexfloat_text(fixed_step);
            write_bounds(output, endpoint_x);
            write_bounds(output, endpoint_y);
            write_bounds(output, tube_x);
            write_bounds(output, tube_y);
            write_bounds(output, prefix_x);
            write_bounds(output, prefix_y);
            output << '\n';
        }
        output.close();

        ofstream summary(argv[2]);
        if (!summary)
        {
            throw runtime_error("cannot open Flow* summary output");
        }
        summary << "{\n"
                << "  \"schema\": \"flowstar_brusselator_second_system_summary_v1\",\n"
                << "  \"fixed_step\": 0.02,\n"
                << "  \"order\": 6,\n"
                << "  \"cutoff\": 1e-10,\n"
                << "  \"target_remainder_radius\": 1e-4,\n"
                << "  \"queue_capacity\": 1000,\n"
                << "  \"requested_horizon\": 20.0,\n"
                << "  \"requested_steps\": " << requested_steps << ",\n"
                << "  \"accepted_steps\": " << step << ",\n"
                << "  \"completed_horizon\": " << decimal17(static_cast<double>(step) * fixed_step) << ",\n"
                << "  \"completed_requested_horizon\": " << (result.isCompleted() && step == requested_steps ? "true" : "false") << ",\n"
                << "  \"result_status_code\": " << result.status << ",\n"
                << "  \"solver_wall_seconds\": " << decimal17(wall_seconds) << ",\n"
                << "  \"reported_core_seconds\": " << decimal17(core_seconds) << "\n"
                << "}\n";
        return result.isCompleted() && step == requested_steps ? 0 : 1;
    }
    catch (const exception &error)
    {
        cerr << error.what() << '\n';
        return 3;
    }
}
