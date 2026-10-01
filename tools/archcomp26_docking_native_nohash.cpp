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

#ifndef DOCKING_PERIODS
#define DOCKING_PERIODS 40
#endif

static Interval radial_margin(const std::vector<Interval> &x)
{
    Interval velocity_squared = x[2].pow(2) + x[3].pow(2);
    Interval position_squared = x[0].pow(2) + x[1].pow(2);
    // Squared sums are mathematically nonnegative, including at zero speed.
    if (velocity_squared.inf() < 0)
        velocity_squared = Interval(0.0, velocity_squared.sup());
    if (position_squared.inf() < 0)
        position_squared = Interval(0.0, position_squared.sup());
    Interval velocity, position;
    velocity_squared.sqrt(velocity);
    position_squared.sqrt(position);
    const Interval threshold = Interval("0.2", "0.2") +
                               Interval("0.002054", "0.002054") * position;
    return velocity - threshold;
}

static bool scan_new_flowpipes(const Result_of_Reachability &result,
                               size_t old_count, int period, std::ofstream &safety)
{
    bool proved = true;
    size_t index = 0;
    for (const Flowpipe &fp : result.flowpipes)
    {
        ++index;
        if (index <= old_count) continue;
        std::vector<Interval> tube;
        fp.tmvPre.intEval(tube, fp.domain);
        if (tube.size() < 4) throw std::runtime_error("short flowpipe tube");
        const Interval q = radial_margin(tube);
        const size_t local = index - old_count;
        safety << period << '\t' << local << '\t' << index << '\t'
               << q.inf() << '\t' << q.sup() << '\n';
        safety.flush();
        if (!safety) throw std::runtime_error("safety output failed");
        if (!std::isfinite(q.inf()) || !std::isfinite(q.sup()) || q.sup() > 0)
            proved = false;
    }
    return proved;
}

int main()
{
    if (!std::getenv("AUTHOR_FIXED_STEPS"))
        throw std::runtime_error("AUTHOR_FIXED_STEPS=1 is required");
    const char *safety_path = std::getenv("DOCKING_SAFETY_LOG");
    if (!safety_path) throw std::runtime_error("DOCKING_SAFETY_LOG is required");
    const char *boxes_path = std::getenv("DOCKING_BOX_LEDGER");
    if (!boxes_path) throw std::runtime_error("DOCKING_BOX_LEDGER is required");
    std::ofstream safety(safety_path);
    if (!safety) throw std::runtime_error("safety output unavailable");
    safety << "period\tlocal_substep\tglobal_substep\tq_lower\tq_upper\n"
           << std::setprecision(17);

    HttpClient httpclient("http://127.0.0.1:5104");
    Client client(httpclient, JSONRPC_CLIENT_V2);
    Variables vars;
    vars.declareVar("sx");
    vars.declareVar("sy");
    vars.declareVar("vx");
    vars.declareVar("vy");
    vars.declareVar("t");
    const int fx_id = vars.declareVar("Fx");
    const int fy_id = vars.declareVar("Fy");
    const unsigned int num_vars = 7;
    const unsigned int num_inputs = 4;
    const unsigned int num_outputs = 2;

    ODE<Real> dynamics({"vx", "vy",
                        "2*0.001027*vy+3*0.001027^2*sx+Fx/12",
                        "-2*0.001027*vx+Fy/12", "1", "0", "0"}, vars);
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.1, 3);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    setting.setRemainderEstimation(std::vector<Interval>(num_vars, Interval(-0.01, 0.01)));

    std::vector<Interval> initial_box = {
        Interval(70, 106), Interval(70, 106),
        Interval(-0.28, 0.28), Interval(-0.28, 0.28),
        Interval(0), Interval(0), Interval(0)
    };
    Flowpipe current(initial_box);
    Symbolic_Remainder symbolic_remainder(current, 1000);
    Result_of_Reachability result;
    arch_ranges::boxes(std::vector<std::vector<Interval>>(1, initial_box),
                       boxes_path);
    const Interval initial_q = radial_margin(initial_box);
    std::cout << std::setprecision(17)
              << "INITIAL_MARGIN " << initial_q.inf() << ' ' << initial_q.sup() << '\n';
    if (initial_q.sup() > 0) {
        std::cout << "UNKNOWN INITIAL_BOX" << std::endl;
        return 2;
    }

    int completed = 0;
    bool all_radial_safe = true;
    for (int period = 0; period < DOCKING_PERIODS; ++period)
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
                const double slope = coefficients["T"][0][j][i].asDouble();
                if (!std::isfinite(slope)) throw std::runtime_error("nonfinite NN slope");
                controls.tms[j] += current.tmvPre.tms[i] * Real(slope);
            }
            const double lo = coefficients["u_min"][0][j].asDouble();
            const double hi = coefficients["u_max"][0][j].asDouble();
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                throw std::runtime_error("invalid NN bias interval");
            controls.tms[j].remainder += Interval(lo, hi);
        }
        current.tmvPre.tms[fx_id] = controls.tms[0];
        current.tmvPre.tms[fy_id] = controls.tms[1];

        const size_t before = result.flowpipes.size();
        const std::vector<Constraint> no_linear_safe_set;
        author_matched::reach(dynamics, result, current, 1, setting,
                              no_linear_safe_set, symbolic_remainder);
        arch_ranges::record(result, 0, 4);
        const bool radial_safe = scan_new_flowpipes(result, before, period, safety);
        all_radial_safe = all_radial_safe && radial_safe;
        const size_t produced = result.flowpipes.size() - before;
        std::cout << "PERIOD " << period << " FLOWPIPES " << produced
                  << " STATUS " << result.status << " RADIAL_SAFE " << radial_safe << '\n';
        if (result.status != COMPLETED_SAFE || produced != 10) {
            std::cout << "COMPLETED_PERIODS " << completed << '/' << DOCKING_PERIODS << '\n'
                      << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << '\n'
                      << "UNKNOWN" << std::endl;
            return 2;
        }
        current = result.fp_end_of_time;
        ++completed;
    }
    std::cout << "COMPLETED_PERIODS " << completed << '/' << DOCKING_PERIODS << '\n'
              << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << '\n'
              << (all_radial_safe ? "VERIFIED" : "UNKNOWN") << std::endl;
    return all_radial_safe ? 0 : 2;
}
