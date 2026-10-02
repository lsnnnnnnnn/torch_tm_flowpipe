#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include "arch_ranges.h"
#include "matched_reach.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace flowstar;
using namespace jsonrpc;

#ifndef BALANCING_PERIODS
#define BALANCING_PERIODS 500
#endif

static bool scan_target_window(const Result_of_Reachability &result,
                               size_t old_count, int period, std::ofstream &target,
                               size_t &checked)
{
    // Checking the closed [8,10] window is equivalent to the fixed repo's
    // (8,10] window for a continuous trajectory and this closed target box.
    if (period < 400) return true;
    bool inside_all = true;
    size_t index = 0;
    for (const Flowpipe &fp : result.flowpipes)
    {
        ++index;
        if (index <= old_count) continue;
        std::vector<Interval> tube;
        fp.tmvPre.intEval(tube, fp.domain);
        if (tube.size() < 5) throw std::runtime_error("short flowpipe tube");
        bool inside = true;
        for (const int state : {0, 2, 3})
            if (!std::isfinite(tube[state].inf()) || !std::isfinite(tube[state].sup()) ||
                tube[state].inf() < -0.001 || tube[state].sup() > 0.001)
                inside = false;
        const size_t local = index - old_count;
        target << period << '\t' << local << '\t' << index << '\t'
               << tube[0].inf() << '\t' << tube[0].sup() << '\t'
               << tube[2].inf() << '\t' << tube[2].sup() << '\t'
               << tube[3].inf() << '\t' << tube[3].sup() << '\t'
               << tube[4].inf() << '\t' << tube[4].sup() << '\t'
               << inside << '\n';
        target.flush();
        if (!target) throw std::runtime_error("target output failed");
        ++checked;
        inside_all = inside_all && inside;
    }
    return inside_all;
}

int main()
{
    if (!std::getenv("AUTHOR_FIXED_STEPS"))
        throw std::runtime_error("AUTHOR_FIXED_STEPS=1 is required");
    const char *target_path = std::getenv("BALANCING_TARGET_LOG");
    if (!target_path) throw std::runtime_error("BALANCING_TARGET_LOG is required");
    const char *boxes_path = std::getenv("BALANCING_BOX_LEDGER");
    if (!boxes_path) throw std::runtime_error("BALANCING_BOX_LEDGER is required");
    std::ofstream target(target_path);
    if (!target) throw std::runtime_error("target output unavailable");
    target << "period\tlocal_substep\tglobal_substep\tx1_lo\tx1_hi\tx3_lo\tx3_hi\tx4_lo\tx4_hi\tt_lo\tt_hi\tinside\n"
           << std::setprecision(17);

    HttpClient httpclient("http://127.0.0.1:5106");
    Client client(httpclient, JSONRPC_CLIENT_V2);
    Variables vars;
    vars.declareVar("x1");
    vars.declareVar("x2");
    vars.declareVar("x3");
    vars.declareVar("x4");
    vars.declareVar("t");
    const int force_id = vars.declareVar("f");
    const unsigned int num_vars = 6;
    const unsigned int num_inputs = 4;
    const unsigned int num_outputs = 1;

    ODE<Real> dynamics({"x2", "2 * f", "x4",
                        "(0.08*0.41*(9.8 * sin(x3) - 2*f * cos(x3)) - 0.0021 * x4) / 0.0105",
                        "1", "0"}, vars);
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.005, 6);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    setting.setRemainderEstimation(std::vector<Interval>(num_vars, Interval(-0.1, 0.1)));

    std::vector<Interval> initial_box = {
        Interval(-0.1, 0.1), Interval(-0.05, 0.05),
        Interval(-0.1, 0.1), Interval(-0.05, 0.05),
        Interval(0), Interval(0)
    };
    Flowpipe current(initial_box);
    Symbolic_Remainder symbolic_remainder(current, 1000);
    Result_of_Reachability result;
    arch_ranges::boxes(std::vector<std::vector<Interval>>(1, initial_box),
                       boxes_path);
    int completed = 0;
    size_t checked = 0;
    bool all_target_inside = true;
    for (int period = 0; period < BALANCING_PERIODS; ++period)
    {
        std::cout << "Step " << period << std::endl;
        Json::Value lower(Json::arrayValue), upper(Json::arrayValue), params, coefficients;
        for (unsigned int i = 0; i < num_inputs; ++i) {
            Interval input;
            current.tmvPre.tms[i].intEval(input, current.domain);
            lower.append(input.inf());
            upper.append(input.sup());
        }
        params["input_lb"] = lower;
        params["input_ub"] = upper;
        client.CallMethod("CROWN_reach", params, coefficients);

        std::vector<Real> zeros(num_outputs, Real(0));
        TaylorModelVec<Real> controls(zeros, num_vars + 1);
        for (unsigned int j = 0; j < num_outputs; ++j) {
            for (unsigned int i = 0; i < num_inputs; ++i) {
                const double slope = coefficients["T"][0][j][i].asFloat();
                if (!std::isfinite(slope)) throw std::runtime_error("nonfinite NN slope");
                controls.tms[j] += current.tmvPre.tms[i] * Real(slope);
            }
            const double lo = coefficients["u_min"][0][j].asFloat();
            const double hi = coefficients["u_max"][0][j].asFloat();
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                throw std::runtime_error("invalid NN bias interval");
            controls.tms[j].remainder += Interval(lo, hi);
        }
        current.tmvPre.tms[force_id] = controls.tms[0];

        const size_t before = result.flowpipes.size();
        const std::vector<Constraint> no_linear_safe_set;
        author_matched::reach(dynamics, result, current, 0.02, setting,
                              no_linear_safe_set, symbolic_remainder);
        arch_ranges::record(result, 0, 5);
        const bool target_inside = scan_target_window(result, before, period, target, checked);
        all_target_inside = all_target_inside && target_inside;
        const size_t produced = result.flowpipes.size() - before;
        std::cout << "PERIOD " << period << " FLOWPIPES " << produced
                  << " STATUS " << result.status << " TARGET_INSIDE " << target_inside << '\n';
        if (result.status != COMPLETED_SAFE || produced != 4) {
            std::cout << "COMPLETED_PERIODS " << completed << '/' << BALANCING_PERIODS << '\n'
                      << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << '\n'
                      << "TARGET_CHECKS " << checked << "/400\n"
                      << "UNKNOWN" << std::endl;
            return 2;
        }
        current = result.fp_end_of_time;
        ++completed;
    }
    std::cout << "COMPLETED_PERIODS " << completed << '/' << BALANCING_PERIODS << '\n'
              << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << '\n'
              << "TARGET_CHECKS " << checked << "/400\n";
    if (BALANCING_PERIODS == 1) {
        std::cout << "COMPLETED_SHORT_PREFIX PROPERTY_NOT_APPLICABLE" << std::endl;
        return 0;
    }
    std::cout << (all_target_inside && checked == 400 ? "VERIFIED_BY_SAVED_BOXES" : "UNKNOWN") << std::endl;
    return all_target_inside && checked == 400 ? 0 : 2;
}
