#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include "arch_ranges.h"
#include "matched_reach.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace jsonrpc;
using namespace std;
using namespace flowstar;

#ifndef UNICYCLE_PERIODS
#define UNICYCLE_PERIODS 1
#endif

// Keep the original Flowpipe(box) center, but cover both requested endpoints.
// The old constructor sized the radius from the upper endpoint only and
// undercovered the requested Unicycle x2 lower endpoint by one binary64 ULP.
static unsigned cover_initial_box(Flowpipe &fp, const vector<Interval> &box,
                                  std::ostream &ledger)
{
    if (intervalNumPrecision != 53 || box.size() != 8 ||
        fp.tmvPre.tms.size() != 8 || fp.tmv.tms.size() != 8 ||
        fp.domain.size() != 9 || fp.domain[0] != Interval(0))
        throw std::runtime_error("initial affine contract differs");
    for (unsigned i = 1; i < 9; ++i)
        if (fp.domain[i] != Interval(-1, 1))
            throw std::runtime_error("initial spatial domain differs");
    unsigned changed = 0;
    ledger << "state\trequested_lo\trequested_hi\tactual_center\tactual_radius\tchanged\n"
           << std::hexfloat;
    for (unsigned i = 0; i < 8; ++i) {
        Real center, old_radius, lo, hi, left, right;
        box[i].toCenterForm(center, old_radius);
        box[i].inf(lo);
        box[i].sup(hi);
        center.sub_RNDU(left, lo);
        hi.sub_RNDU(right, center);
        const Real radius = left > right ? left : right;
        if (radius < old_radius || radius < 0)
            throw std::runtime_error("initial radius shrank");
        const auto &old = fp.tmvPre.tms[i];
        if (old.remainder != Interval(0))
            throw std::runtime_error("initial remainder differs");
        Real actual_center(0), actual_radius(0);
        bool seen_center = false, seen_radius = false;
        for (const auto &term : old.expansion.terms) {
            const auto &degrees = term.adoptionDegrees();
            if (degrees.size() != 9)
                throw std::runtime_error("initial monomial dimension differs");
            unsigned total = 0;
            for (auto degree : degrees) total += degree;
            if (total == 0 && !seen_center) {
                seen_center = true;
                actual_center = term.adoptionCoefficient();
            } else if (total == 1 && degrees[i+1] == 1 && !seen_radius) {
                seen_radius = true;
                actual_radius = term.adoptionCoefficient();
            } else {
                throw std::runtime_error("initial affine support differs");
            }
        }
        if (actual_center != center || actual_radius != old_radius)
            throw std::runtime_error("initial constructor coefficient differs");
        if (radius != old_radius) {
            vector<Real> coefficients(9, Real(0));
            coefficients[i+1] = radius;
            TaylorModel<Real> replacement(coefficients, Interval(0));
            replacement += TaylorModel<Real>(center, 9);
            fp.tmvPre.tms[i] = replacement;
            ++changed;
        }
        const auto &actual = fp.tmvPre.tms[i];
        if (actual.remainder != Interval(0))
            throw std::runtime_error("repaired initial remainder differs");
        Real after_center(0), after_radius(0);
        seen_center = seen_radius = false;
        for (const auto &term : actual.expansion.terms) {
            const auto &degrees = term.adoptionDegrees();
            if (degrees.size() != 9)
                throw std::runtime_error("repaired monomial dimension differs");
            unsigned total = 0;
            for (auto degree : degrees) total += degree;
            if (total == 0 && !seen_center) {
                seen_center = true;
                after_center = term.adoptionCoefficient();
            } else if (total == 1 && degrees[i+1] == 1 && !seen_radius) {
                seen_radius = true;
                after_radius = term.adoptionCoefficient();
            } else {
                throw std::runtime_error("repaired initial affine support differs");
            }
        }
        if (after_center != center || after_radius != radius)
            throw std::runtime_error("repaired initial coefficients differ");
        ledger << i << '\t' << lo.toDouble() << '\t' << hi.toDouble()
               << '\t' << after_center.toDouble() << '\t' << after_radius.toDouble()
               << '\t' << (radius != old_radius) << '\n';
    }
    ledger.flush();
    if (!ledger) throw std::runtime_error("initial affine ledger failed");
    return changed;
}


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5107");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 8;
    unsigned int num_nn_input = 4;
    unsigned int num_nn_output = 2;
    Variables vars;
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int t_id = vars.declareVar("t");
    int u1_id = vars.declareVar("u1");
    int u2_id = vars.declareVar("u2");
    int w_id = vars.declareVar("w");

    ODE<Real> dynamics({"x4 * cos(x3)",
                        "x4 * sin(x3)",
                        "u2 - 20",
                        "u1 + w - 20",
                        "1",
                        "0",
                        "0",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.02, 2);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = UNICYCLE_PERIODS;
    Interval init_x1(9.5, 9.55), init_x2(-4.5, -4.45), init_x3(2.1, 2.11), init_x4(1.5, 1.51),
             init_t(0), init_u1(0), init_u2(0), init_w(-1e-4, 1e-4);
    vector<Interval> X0;
    X0.push_back(init_x1);
    X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_x4);
    X0.push_back(init_t);
    X0.push_back(init_u1);
    X0.push_back(init_u2);
    X0.push_back(init_w);

    // Translate the initial set to a flowpipe
    Flowpipe initial_set(X0);
    const char *affine_path = std::getenv("UNICYCLE_INITIAL_AFFINE_LOG");
    const char *boxes_path = std::getenv("UNICYCLE_BOX_LOG");
    if (!affine_path || !boxes_path)
        throw std::runtime_error("initial evidence paths are required");
    std::ofstream affine(affine_path);
    if (!affine) throw std::runtime_error("initial affine ledger unavailable");
    const unsigned repaired = cover_initial_box(initial_set, X0, affine);
    arch_ranges::boxes(std::vector<std::vector<Interval>>{X0}, boxes_path);
    cout << "INITIAL_AFFINE_REPAIRED " << repaired << "/8" << endl;
    if (argc == 2 && std::string(argv[1]) == "--preflight")
        return 0;
    if (argc != 1) throw std::runtime_error("unknown native argument");
    Symbolic_Remainder symbolic_remainder(initial_set, 1000);
    vector<Constraint> safeSet;
    Result_of_Reachability result;

    int final_result = 0;
    auto start = std::chrono::steady_clock::now();
    int completed_periods = 0;

    for (int iter = 0; iter < steps; iter++)
    {
        cout << "Step " << iter << endl;
        
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int i = 0; i < num_nn_input; i++)
        {
            Interval input_range_temp;
            initial_set.tmvPre.tms[i].intEval(input_range_temp, initial_set.domain);
            input_lb.append(input_range_temp.inf());
            input_ub.append(input_range_temp.sup());
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        cl.CallMethod("CROWN_reach", params, output_coefficients);
        // Unpack results from CROWN
        Matrix<Real> T(num_nn_output, num_nn_input, Real(0));
        vector<Real> c_vector;
        vector<double> interval_r;
        for (int j = 0; j < num_nn_output; j++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                T[j][i] = output_coefficients["T"][j][i].asFloat();
            }
            double u_max = output_coefficients["u_max"][j].asFloat();
            double u_min = output_coefficients["u_min"][j].asFloat();
            c_vector.push_back((u_max + u_min) / 2);
            interval_r.push_back((u_max - u_min) / 2);
        }
        
        // Construct new Taylor Models
        TaylorModelVec<Real> tmv_output(c_vector, numVars);
        for (int j = 0; j < num_nn_output; j++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                tmv_output.tms[j] += initial_set.tmvPre.tms[i] * T[j][i];
            }
            Interval remainder_temp(-interval_r[j], interval_r[j]);
            tmv_output.tms[j].remainder += remainder_temp;
        }

        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];

        // Flow*
        author_matched::reach(dynamics, result, initial_set, 0.2, setting, safeSet, symbolic_remainder);
        arch_ranges::record(result,0,4);
        const size_t segments = result.flowpipes.size();
        cout << "PERIOD " << iter << " STATUS " << result.status
             << " CUMULATIVE_SEGMENTS " << segments << endl;
        
        if ((result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE ||
             result.status == COMPLETED_UNKNOWN) && segments == size_t((iter+1)*10))
        {
            initial_set = result.fp_end_of_time;
            ++completed_periods;
        }
        else
        {
            cout << "Flow* terminated." << endl;
            final_result = 2;
            break;
        }
    }

    cout << "COMPLETED_PERIODS " << completed_periods << '/' << steps << endl;
    cout << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << endl;
    if (final_result != 0) {
        cout << "UNKNOWN" << endl;
        return 2;
    }
    if (steps == 1) {
        cout << "COMPLETED_SHORT_PREFIX PROPERTY_NOT_APPLICABLE" << endl;
        return 0;
    }

    // A terminal inclusion is sufficient for reach sometime within 10 s.
    vector<Constraint> targetSet;
    vector<string> constraints = {"x1 - 0.6", "-x1 - 0.6",
                                  "x2 - 0.2", "-x2 - 0.2",
                                  "x3 - 0.06", "-x3 - 0.06",
                                  "x4 - 0.3", "-x4 - 0.3"};
    for (int i = 0; i < 8; i++)
    {
        Constraint c_temp(constraints[i], vars);
        targetSet.push_back(c_temp);
    }

    bool b = result.fp_end_of_time.isInTarget(targetSet, setting);

    if (b && final_result == 0)
    {
        cout << "VERIFIED" << endl;
    }
    else
    {
        cout << "UNKNOWN" << endl;
    }


    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("time cost: %lf\n", (double)(duration.count() / 1000.0));

    // result.transformToTaylorModels(setting);
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("x1", "x2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "Unicycle_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}
