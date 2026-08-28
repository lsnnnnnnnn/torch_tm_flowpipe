#include "Continuous.h"
#include "CausalTrace.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <list>
#include <stdexcept>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

vector<Interval> endpoint_table(const Taylor_Model_Setting &setting)
{
    vector<Interval> result;
    for (size_t index = 0; index < setting.step_end_exp_table.size(); ++index)
    {
        result.push_back(Interval(setting.step_end_exp_table[index]));
    }
    return result;
}

void print_interval(const char *name, const Interval &value)
{
    cout << "\"" << name << "\":{";
    cout << "\"lo\":" << setprecision(17) << value.inf() << ',';
    cout << "\"hi\":" << setprecision(17) << value.sup() << ',';
    cout << "\"lo_hex\":\"" << hexfloat << value.inf() << "\",";
    cout << "\"hi_hex\":\"" << hexfloat << value.sup() << "\"}" << defaultfloat;
}

} // namespace

int main()
{
    try
    {
        const double fixed_step = 0.02;
        const unsigned int order = 6;
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
        setting.setRemainderEstimation(
            vector<Interval>(variables.size(), Interval(-1e-4, 1e-4)));
        vector<Interval> box(variables.size());
        box[x_id] = Interval("1.48", "1.52");
        box[y_id] = Interval("2.98", "3.02");
        Flowpipe initial(box);
        Symbolic_Remainder symbolic_remainder(initial, 1000);
        vector<Constraint> safe_set;
        Result_of_Reachability result;
        ode.reach(result, initial, fixed_step, setting, safe_set, symbolic_remainder);
        if (result.flowpipes.size() != 1)
        {
            throw runtime_error("one-step diagnostic did not publish exactly one flowpipe");
        }

        const Flowpipe &accepted = result.flowpipes.front();
        const vector<Interval> endpoint = endpoint_table(setting.tm_setting);
        TaylorModelVec<Real> endpoint_composed;
        TaylorModelVec<Real> tube_composed;
        accepted.compose_normal(
            endpoint_composed,
            endpoint,
            setting.tm_setting.order,
            setting.tm_setting.cutoff_threshold);
        accepted.compose_normal(
            tube_composed,
            setting.tm_setting.step_exp_table,
            setting.tm_setting.order,
            setting.tm_setting.cutoff_threshold);
        flowstar_causal::set_step_context(0, 0.0, fixed_step);
        flowstar_causal::emit_tmv("driver_endpoint_composed", endpoint_composed);
        flowstar_causal::emit_tmv("driver_tube_composed", tube_composed);

        vector<Interval> endpoint_box;
        vector<Interval> tube_box;
        accepted.intEvalNormal(
            endpoint_box,
            endpoint,
            setting.tm_setting.order,
            setting.tm_setting.cutoff_threshold);
        accepted.intEvalNormal(
            tube_box,
            setting.tm_setting.step_exp_table,
            setting.tm_setting.order,
            setting.tm_setting.cutoff_threshold);
        if (endpoint_box.size() < 2 || tube_box.size() < 2)
        {
            throw runtime_error("Flow* returned fewer than two plant dimensions");
        }
        cout << '{';
        print_interval("endpoint_x", endpoint_box[0]);
        cout << ',';
        print_interval("endpoint_y", endpoint_box[1]);
        cout << ',';
        print_interval("tube_x", tube_box[0]);
        cout << ',';
        print_interval("tube_y", tube_box[1]);
        cout << "}\n";
        return 0;
    }
    catch (const exception &error)
    {
        cerr << error.what() << '\n';
        return 2;
    }
}
