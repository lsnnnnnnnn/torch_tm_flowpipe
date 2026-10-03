#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include "arch_ranges.h"
#include "matched_reach.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <chrono>
using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5105");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 19;
    unsigned int num_nn_input = 12;
    unsigned int num_nn_output = 6;
    Variables vars;
    int x_id = vars.declareVar("x");
    int y_id = vars.declareVar("y");
    int z_id = vars.declareVar("z");
    int u_id = vars.declareVar("u");
    int v_id = vars.declareVar("v");
    int w_id = vars.declareVar("w");
    int phi_id = vars.declareVar("phi");
    int theta_id = vars.declareVar("theta");
    int psi_id = vars.declareVar("psi");
    int r_id = vars.declareVar("r");
    int p_id = vars.declareVar("p");
    int q_id = vars.declareVar("q");
    int t_id = vars.declareVar("t");
    int Fx_id = vars.declareVar("Fx");
    int Fy_id = vars.declareVar("Fy");
    int Fz_id = vars.declareVar("Fz");
    int Mx_id = vars.declareVar("Mx");
    int My_id = vars.declareVar("My");
    int Mz_id = vars.declareVar("Mz");

    ODE<Real> dynamics({"cos(psi)*cos(theta) * u + (-sin(psi)*cos(phi) + cos(psi)*sin(theta)*sin(phi)) * v + (sin(psi)*sin(phi) + cos(psi)*sin(theta)*cos(phi)) * w",
                        "sin(psi)*cos(theta) * u + (cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi)) * v + (-cos(psi)*sin(phi) + sin(psi)*sin(theta)*cos(phi)) * w",
                        "-sin(theta) * u + cos(theta)*sin(phi) * v + cos(theta)*cos(phi) * w",
                        "-sin(theta) + Fx - q * w + r * v",
                        "cos(theta) * sin(phi) + Fy - r * u + p * w",
                        "cos(theta) * cos(phi) + Fz - p * v + q * u",
                        "(cos(theta) * p + sin(theta)*sin(phi) * q + sin(theta)*cos(phi) * r) / cos(theta)",
                        "(cos(theta)*cos(phi) * q - cos(theta) * sin(phi) * r) / cos(theta)",
                        "(sin(phi) * q + cos(phi) * r) / cos(theta)",
                        "Mz",
                        "Mx",
                        "My",
                        "1", "0", "0", "0", "0", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.01, 3);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    const char *bits_env = std::getenv("AIRPLANE_CELL_BITS");
    if (!bits_env) throw std::runtime_error("AIRPLANE_CELL_BITS required");
    const std::string bits(bits_env);
    if (bits.size() != 6 || bits.find_first_not_of("01") != std::string::npos)
        throw std::runtime_error("invalid AIRPLANE_CELL_BITS");
    auto lo = [&](size_t i) { return bits[i] == '1' ? 0.5 : 0.0; };
    auto hi = [&](size_t i) { return bits[i] == '1' ? 1.0 : 0.5; };

    // Initial set
    int steps = 1;
    Interval init_x(0), init_y(0), init_z(0),
             init_u(lo(0), hi(0)), init_v(lo(1), hi(1)), init_w(lo(2), hi(2)),
             init_phi(lo(3), hi(3)), init_theta(lo(4), hi(4)), init_psi(lo(5), hi(5)),
             init_r(0), init_p(0), init_q(0),
             init_t(0),
             init_Fx(0), init_Fy(0), init_Fz(0),
             init_Mx(0), init_My(0), init_Mz(0);
    vector<Interval> X0;
    X0.push_back(init_x);
    X0.push_back(init_y);
    X0.push_back(init_z);
    X0.push_back(init_u);
    X0.push_back(init_v);
    X0.push_back(init_w);
    X0.push_back(init_phi);
    X0.push_back(init_theta);
    X0.push_back(init_psi);
    X0.push_back(init_r);
    X0.push_back(init_p);
    X0.push_back(init_q);
    X0.push_back(init_t);
    X0.push_back(init_Fx);
    X0.push_back(init_Fy);
    X0.push_back(init_Fz);
    X0.push_back(init_Mx);
    X0.push_back(init_My);
    X0.push_back(init_Mz);

    // Translate the initial set to a flowpipe
    Flowpipe initial_set(X0);
    Symbolic_Remainder symbolic_remainder(initial_set, 1000);

    vector<string> constraints = {"-y - 1", "y - 1",
                                  "-phi - 1", "phi - 1",
                                  "-theta - 1", "theta - 1",
                                  "-psi - 1", "psi - 1"};

    vector<Constraint> safeSet;
    for (int i = 0; i < 8; i++)
    {
        Constraint c_temp(constraints[i], vars);
        safeSet.push_back(c_temp);
    }
    Result_of_Reachability result;
    int final_result = 0;
    int completed_periods = 0;
    const char *check_path = std::getenv("AIRPLANE_CHECK_LOG");
    if (!check_path) throw std::runtime_error("AIRPLANE_CHECK_LOG required");
    std::ofstream checks(check_path);
    if (!checks) throw std::runtime_error("check log unavailable");
    checks << "period\tlocal_substep\tglobal_substep\ty_lo\ty_hi\tphi_lo\tphi_hi\ttheta_lo\ttheta_hi\tpsi_lo\tpsi_hi\tcos_theta_lo\tcos_theta_hi\n" << std::setprecision(17);

    if (argc==2) { arch_ranges::boxes(std::vector<std::vector<Interval>>{X0},argv[1]); return 0; }
    auto start = std::chrono::steady_clock::now();

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
        vector<Real> zeros(num_nn_output, Real(0));
        TaylorModelVec<Real> tmv_output(zeros, numVars);
        for (unsigned int j = 0; j < num_nn_output; ++j) {
            for (unsigned int i = 0; i < num_nn_input; ++i) {
                double slope = output_coefficients["T"][j][i].asDouble();
                if (!std::isfinite(slope)) throw std::runtime_error("nonfinite NN slope");
                tmv_output.tms[j] += initial_set.tmvPre.tms[i] * Real(slope);
            }
            double lo = output_coefficients["u_min"][j].asDouble();
            double hi = output_coefficients["u_max"][j].asDouble();
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                throw std::runtime_error("invalid NN bias interval");
            tmv_output.tms[j].remainder += Interval(lo, hi);
        }

        initial_set.tmvPre.tms[Fx_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[Fy_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[Fz_id] = tmv_output.tms[2];
        initial_set.tmvPre.tms[Mx_id] = tmv_output.tms[3];
        initial_set.tmvPre.tms[My_id] = tmv_output.tms[4];
        initial_set.tmvPre.tms[Mz_id] = tmv_output.tms[5];

        // Flow*: keep the full local tube and stop on any non-safe or incomplete step.
        size_t prior = result.flowpipes.size();
        author_matched::reach(dynamics, result, initial_set, 0.01, setting, safeSet, symbolic_remainder);
        arch_ranges::record(result, 0, 12);
        bool independently_box_safe = true;
        size_t index = 0;
        const int safe_ids[] = {y_id, phi_id, theta_id, psi_id};
        for (const Flowpipe &fp : result.flowpipes) {
            ++index;
            if (index <= prior) continue;
            vector<Interval> tube;
            fp.tmvPre.intEval(tube, fp.domain);
            if (tube.size() < 12) throw std::runtime_error("short Airplane tube");
            Interval cosine = tube[theta_id].cos();
            checks << iter << '\t' << (index - prior) << '\t' << index;
            for (int id : safe_ids) {
                checks << '\t' << tube[id].inf() << '\t' << tube[id].sup();
                if (tube[id].inf() < -1 || tube[id].sup() > 1) independently_box_safe = false;
            }
            checks << '\t' << cosine.inf() << '\t' << cosine.sup() << '\n';
            if (cosine.inf() <= 0) independently_box_safe = false;
        }
        checks.flush();
        if (!checks) throw std::runtime_error("Airplane check output failed");
        size_t produced = result.flowpipes.size() - prior;
        cout << "PERIOD " << iter << " FLOWPIPES " << produced
             << " STATUS " << result.status << " BOX_SAFE " << independently_box_safe << endl;
        if (result.status != COMPLETED_SAFE || produced != 1 || !independently_box_safe) {
            final_result = 2;
            break;
        }
        initial_set = result.fp_end_of_time;
        ++completed_periods;
    }
    cout << "DIAGNOSTIC_COMPLETED_CALLS " << completed_periods << "/" << steps << endl;
    cout << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << endl;
    if (final_result == 0)
    {
        cout << "ONE_STEP_NUMERIC_ACCEPTED_ONLY" << endl;
    }
    else if (final_result == 1)
    {
        cout << "FALSIFIED" << endl;
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
    // plot_setting.setOutputDims("y", "phi");
    // plot_setting.plot_2D_octagon_MATLAB("./", "airplane_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return final_result == 0 ? 0 : 2;
}