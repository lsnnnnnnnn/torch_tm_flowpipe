#include "Continuous.h"
#include "quad_gate_observer.h"
#include "arch_ranges.h"
#include "matched_reach.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <boost/thread/thread.hpp>
#include <chrono>

// Temporarily undefine PN to avoid conflict with Boost Asio
#ifdef PN
#undef PN
#endif
#include <boost/asio.hpp>

// Redefine PN after Boost Asio headers are included
#ifdef PN_CONFLICT
#define PN PN_CONFLICT
#else
#define PN 1
#endif

using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *argv[])
{
    // Isolated saved-RPC controller construction: no server or RPC connection.

    // Declaration of variables
    unsigned int numVars = 16;
    unsigned int num_nn_input = 12;
    unsigned int num_nn_output = 3;
    Variables vars;
    int x_ids[num_nn_input];
    for (int i = 1; i <= num_nn_input; i++)
    {
        string var_name = "x" + to_string(i);
        int var_id = vars.declareVar(var_name);
        x_ids[i-1] = var_id;
    }
    int t_id = vars.declareVar("t");
    int u_ids[num_nn_output];
    for (int i = 1; i <= num_nn_output; i++)
    {
        string var_name = "u" + to_string(i);
        int var_id = vars.declareVar(var_name);
        u_ids[i-1] = var_id;
    }


    ODE<Real> dynamics({"cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6",
                        "cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6",
                        "sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6",
                        "x12*x5 - x11*x6 - 9.81*sin(x8)",
                        "x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)",
                        "x11*x4 - x10*x5 + 9.81 *cos(x8)*cos(x7) - 9.81 - u1 / 1.4",
                        "x10 + sin(x7)*sin(x8)/cos(x8)*x11 + cos(x7)*sin(x8)/cos(x8)*x12",
                        "cos(x7)*x11 - sin(x7)*x12",
                        "sin(x7)*x11/cos(x8) - cos(x7)*x12/cos(x8)",
                        "x11*x12*(0.054 - 0.104) / 0.054 + u2 / 0.054",
                        "(0.104 - 0.054)*x10*x12 / 0.054 + u3 / 0.054",
                        "0",
                        "1", "0", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.005, 2);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.1, 0.1);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);
    vector<Constraint> safeSet;

    // Initial set
    int steps = 1;
    vector<Interval> X0;
    Interval init_x1(-0.4, 0.4), init_x2(-0.4, 0.4), init_x3(-0.4, 0.4),
             init_x4(-0.4, 0.4), init_x5(-0.4, 0.4), init_x6(-0.4, 0.4),
             init_x7(0), init_x8(0), init_x9(0), init_x10(0), init_x11(0), init_x12(0),
             init_t(0), init_u1(0), init_u2(0), init_u3(0);

    list<Interval> list_x1;
    init_x1.split(list_x1, 8);
    list<Interval> list_x2;
    init_x2.split(list_x2, 8);
    list<Interval> list_x3;
    init_x3.split(list_x3, 8);
    list<Interval> list_x4;
    init_x4.split(list_x4, 2);
    list<Interval> list_x5;
    init_x5.split(list_x5, 1);
    list<Interval> list_x6;
    init_x6.split(list_x6, 1);

    vector<Flowpipe> initial_sets;
    vector<vector<Interval> > initial_boxes;
    for (auto iter1 = list_x1.begin(); iter1 != list_x1.end(); ++iter1)
    {
        for (auto iter2 = list_x2.begin(); iter2 != list_x2.end(); ++iter2)
        {
            for (auto iter3 = list_x3.begin(); iter3 != list_x3.end(); ++iter3)
            {
                for (auto iter4 = list_x4.begin(); iter4 != list_x4.end(); ++iter4)
                {
                    for (auto iter5 = list_x5.begin(); iter5 != list_x5.end(); ++iter5)
                    {
                        for (auto iter6 = list_x6.begin(); iter6 != list_x6.end(); ++iter6)
                        {
                            vector<Interval> X0;
                            X0.push_back(*iter1);
                            X0.push_back(*iter2);
                            X0.push_back(*iter3);
                            X0.push_back(*iter4);
                            X0.push_back(*iter5);
                            X0.push_back(*iter6);
                            X0.push_back(init_x7);
                            X0.push_back(init_x8);
                            X0.push_back(init_x9);
                            X0.push_back(init_x10);
                            X0.push_back(init_x11);
                            X0.push_back(init_x12);
                            X0.push_back(init_t);
                            X0.push_back(init_u1);
                            X0.push_back(init_u2);
                            X0.push_back(init_u3);
                            Flowpipe initial_set_temp(X0);
                            initial_sets.push_back(initial_set_temp);
                            initial_boxes.push_back(X0);
                        }
                    }
                }
            }
        }
    }


    if (initial_sets.size() != 1024 || initial_boxes.size() != 1024)
        throw std::runtime_error("unexpected source partition size");
    {
        std::ofstream boxes(std::string(quad_gate::directory()) + "/source_boxes.csv");
        boxes << std::setprecision(17) << "lane,coord,lo,hi\n";
        for (unsigned lane = 0; lane < initial_boxes.size(); ++lane)
            for (unsigned coord = 0; coord < 12; ++coord)
                boxes << lane << ',' << coord + 1 << ','
                      << initial_boxes[lane][coord].inf() << ','
                      << initial_boxes[lane][coord].sup() << '\n';
    }
    vector<Symbolic_Remainder> symbolic_remainders;
    vector<Result_of_Reachability> results;
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        Flowpipe initial_set = initial_sets[sub_iter];
        Symbolic_Remainder symbolic_remainder_temp(initial_set, 1000);
        symbolic_remainders.push_back(symbolic_remainder_temp);
        Result_of_Reachability result_temp;
        results.push_back(result_temp);
    }

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < steps; iter++)
    {
        cout << "Step " << iter << endl;
        cout << "Constructing input." << endl;
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                Interval input_range_temp;
                initial_sets[sub_iter].tmvPre.tms[i].intEval(input_range_temp, initial_sets[sub_iter].domain);
                input_lb.append(input_range_temp.inf());
                input_ub.append(input_range_temp.sup());
            }
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        cout << "Loading saved first CROWN batch and corrected residual endpoints." << endl;
        const char *old_path = std::getenv("QUAD_OLD_RPC");
        const char *correction_path = std::getenv("QUAD_CORRECTED_BIASES");
        if (!old_path || !correction_path) throw std::runtime_error("input paths missing");
        Json::Value saved, correction;
        std::ifstream old_stream(old_path), correction_stream(correction_path);
        if (!old_stream || !correction_stream) throw std::runtime_error("cannot open saved inputs");
        old_stream >> saved;
        correction_stream >> correction;
        if (saved["params"]["input_lb"].size() != input_lb.size() ||
            saved["params"]["input_ub"].size() != input_ub.size() ||
            correction["u_min"].size() != 1024 || correction["u_max"].size() != 1024)
            throw std::runtime_error("saved batch cardinality mismatch");
        for (unsigned pos = 0; pos < input_lb.size(); ++pos) {
            if (saved["params"]["input_lb"][pos].asDouble() != input_lb[pos].asDouble() ||
                saved["params"]["input_ub"][pos].asDouble() != input_ub[pos].asDouble()) {
                cerr << "FIRST_REFUSAL_INPUT_MISMATCH pos=" << pos << endl;
                return 3;
            }
        }
        output_coefficients = saved["coefficients"];
        output_coefficients["u_min"] = correction["u_min"];
        output_coefficients["u_max"] = correction["u_max"];
        quad_gate::save_rpc(params, output_coefficients);
        std::ofstream control_trace(std::string(quad_gate::directory()) + "/construction.csv");
        if (!control_trace) throw std::runtime_error("cannot open construction receipt");
        control_trace << "lane,output,corrected_lower,corrected_upper,center,radius,"
                         "residual_lower,residual_upper,control_lower,control_upper,"
                         "affine_reference_lower,affine_reference_upper\n";
        control_trace << std::setprecision(17);
        if (input_lb.size() != 1024 * num_nn_input ||
            input_ub.size() != 1024 * num_nn_input ||
            output_coefficients["T"].size() != 1024)
            throw std::runtime_error("CROWN batch cardinality mismatch");
        for (unsigned lane = 0; lane < initial_boxes.size(); ++lane) {
            for (unsigned coord = 0; coord < 12; ++coord) {
                const unsigned pos = lane * 12 + coord;
                if (input_lb[pos].asDouble() > initial_boxes[lane][coord].inf() ||
                    input_ub[pos].asDouble() < initial_boxes[lane][coord].sup()) {
                    cerr << "RPC_INPUT_OMITS_SOURCE lane=" << lane
                         << " coord=" << coord + 1 << endl;
                    return 3;
                }
            }
        }
        cout << "RPC_COVERS_ALL_SOURCE_BOXES" << endl;
        // Unpack results from CROWN
        cout << "Unpacking output from CROWN." << endl;
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            Matrix<Real> T(num_nn_output, num_nn_input, Real(0));
            vector<Real> c_vector;
            vector<double> interval_r;
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    T[j][i] = output_coefficients["T"][sub_iter][j][i].asFloat();
                }
                double u_max = output_coefficients["u_max"][sub_iter][j].asFloat();
                double u_min = output_coefficients["u_min"][sub_iter][j].asFloat();
                c_vector.push_back((u_max + u_min) / 2);
                interval_r.push_back((u_max - u_min) / 2);
            }
            
            // Construct new Taylor Models
            TaylorModelVec<Real> tmv_output(c_vector, numVars + 1);
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    tmv_output.tms[j] += initial_sets[sub_iter].tmvPre.tms[i] * T[j][i];
                }
                Interval remainder_temp(-interval_r[j], interval_r[j]);
                tmv_output.tms[j].remainder += remainder_temp;

                const double corrected_lower = output_coefficients["u_min"][sub_iter][j].asFloat();
                const double corrected_upper = output_coefficients["u_max"][sub_iter][j].asFloat();
                Interval residual_control(c_vector[j]);
                residual_control += remainder_temp;
                Interval control_range;
                tmv_output.tms[j].intEval(control_range, initial_sets[sub_iter].domain);
                Interval affine_reference(corrected_lower, corrected_upper);
                for (int i = 0; i < num_nn_input; ++i) {
                    const unsigned pos = sub_iter * num_nn_input + i;
                    Interval rpc_input(input_lb[pos].asDouble(), input_ub[pos].asDouble());
                    affine_reference += rpc_input * T[j][i];
                }
                const double residual_lower = residual_control.inf();
                const double residual_upper = residual_control.sup();
                const double control_lower = control_range.inf();
                const double control_upper = control_range.sup();
                const double reference_lower = affine_reference.inf();
                const double reference_upper = affine_reference.sup();
                control_trace << sub_iter << ',' << j + 1 << ',' << corrected_lower << ','
                    << corrected_upper << ',' << c_vector[j] << ',' << interval_r[j] << ','
                    << residual_lower << ',' << residual_upper << ',' << control_lower << ','
                    << control_upper << ',' << reference_lower << ',' << reference_upper << '\n';
                control_trace.flush();
                if (!std::isfinite(residual_lower) || !std::isfinite(residual_upper) ||
                    !std::isfinite(control_lower) || !std::isfinite(control_upper) ||
                    residual_lower > corrected_lower || residual_upper < corrected_upper ||
                    control_lower > reference_lower || control_upper < reference_upper) {
                    cerr << "FIRST_REFUSAL_CONTROL_CONSTRUCTION lane=" << sub_iter
                         << " output=" << j + 1 << endl;
                    return 4;
                }
            }

            for (int j = 0; j < num_nn_output; j++)
            {
                initial_sets[sub_iter].tmvPre.tms[u_ids[j]] = tmv_output.tms[j];
            }
        }


        cout << "CONTROL_CONSTRUCTION_ACCEPTED_3072_NO_ODE" << endl;
        return 0;
    }
}
