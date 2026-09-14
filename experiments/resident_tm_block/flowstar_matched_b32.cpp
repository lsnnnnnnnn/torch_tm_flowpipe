// Matched-workload Flow* timing harness for the resident TM block study.
//
// This file is deliberately only a driver: it does not modify Flow* algorithms.
// It runs all 8x4 outward-binary64 partition boxes sequentially in one process.

#include "Continuous.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flowstar;
using namespace std;

namespace {

using Clock = chrono::steady_clock;

struct Axis {
    array<const char *, 8> x_lo;
    array<const char *, 8> x_hi;
    array<const char *, 4> y_lo;
    array<const char *, 4> y_hi;
};

const Axis &axes_for(const string &plant) {
    // These are the exact outward binary64 endpoints from PARTITION_PLAN.json.
    static const Axis vdp = {{
        "0x1.1999999999999p+0", "0x1.2333333333333p+0",
        "0x1.2ccccccccccccp+0", "0x1.3666666666666p+0",
        "0x1.4000000000000p+0", "0x1.4999999999999p+0",
        "0x1.5333333333333p+0", "0x1.5ccccccccccccp+0",
    }, {
        "0x1.2333333333334p+0", "0x1.2cccccccccccdp+0",
        "0x1.3666666666667p+0", "0x1.4000000000000p+0",
        "0x1.499999999999ap+0", "0x1.5333333333334p+0",
        "0x1.5cccccccccccdp+0", "0x1.6666666666667p+0",
    }, {
        "0x1.2ccccccccccccp+1", "0x1.3000000000000p+1",
        "0x1.3333333333333p+1", "0x1.3666666666666p+1",
    }, {
        "0x1.3000000000000p+1", "0x1.3333333333334p+1",
        "0x1.3666666666667p+1", "0x1.399999999999ap+1",
    }};
    static const Axis bruss = {{
        "0x1.7ae147ae147aep+0", "0x1.7c28f5c28f5c2p+0",
        "0x1.7d70a3d70a3d7p+0", "0x1.7eb851eb851ebp+0",
        "0x1.8000000000000p+0", "0x1.8147ae147ae14p+0",
        "0x1.828f5c28f5c28p+0", "0x1.83d70a3d70a3dp+0",
    }, {
        "0x1.7c28f5c28f5c3p+0", "0x1.7d70a3d70a3d8p+0",
        "0x1.7eb851eb851ecp+0", "0x1.8000000000000p+0",
        "0x1.8147ae147ae15p+0", "0x1.828f5c28f5c29p+0",
        "0x1.83d70a3d70a3ep+0", "0x1.851eb851eb852p+0",
    }, {
        "0x1.7d70a3d70a3d7p+1", "0x1.7eb851eb851ebp+1",
        "0x1.8000000000000p+1", "0x1.8147ae147ae14p+1",
    }, {
        "0x1.7eb851eb851ecp+1", "0x1.8000000000000p+1",
        "0x1.8147ae147ae15p+1", "0x1.828f5c28f5c29p+1",
    }};
    return plant == "brusselator" ? bruss : vdp;
}

double parse_hex(const char *text) {
    char *end = nullptr;
    const double value = strtod(text, &end);
    if (end == text || *end != '\0') {
        throw runtime_error(string("invalid hexadecimal endpoint: ") + text);
    }
    return value;
}

double seconds(const Clock::time_point &begin, const Clock::time_point &end) {
    return chrono::duration<double>(end - begin).count();
}

void write_interval(ostream &out, const Interval &value) {
    out << '[' << setprecision(17) << value.inf() << ',' << value.sup() << ']';
}

}  // namespace

int main(int argc, char **argv) {
    const auto main_start = Clock::now();
    if (argc != 5) {
        cerr << "usage: flowstar_matched_b32 brusselator|van_der_pol STEPS "
                "TASKS_JSONL SUMMARY_JSON\n";
        return 2;
    }

    try {
        const string plant = argv[1];
        const bool bruss = plant == "brusselator";
        if (!bruss && plant != "van_der_pol") {
            throw runtime_error("unknown plant");
        }
        const int requested_steps = stoi(argv[2]);
        if (requested_steps <= 0) {
            throw runtime_error("STEPS must be positive");
        }

        ofstream tasks(argv[3]);
        if (!tasks) {
            throw runtime_error("task output unavailable");
        }
        tasks << setprecision(17);

        const double h = bruss ? 0.02 : 0.01;
        const int order = bruss ? 6 : 4;
        const size_t sr_capacity = bruss ? 1000 : 100;
        const Axis &axes = axes_for(plant);

        const auto setup_start = Clock::now();
        Variables vars;
        vars.declareVar("x");
        vars.declareVar("y");
        const vector<string> rhs = bruss
            ? vector<string>{"1 + x*(x*y - 4)", "x*(3 - x*y)"}
            : vector<string>{"y", "y - x - x*x*y"};
        ODE<Real> ode(rhs, vars);
        Computational_Setting setting(vars);
        setting.printOff();
        if (!setting.setFixedStepsize(h, order)) {
            throw runtime_error("invalid fixed step");
        }
        setting.setCutoffThreshold(1e-10);
        setting.setRemainderEstimation(
            vector<Interval>(2, Interval(-1e-4, 1e-4)));
        vector<Constraint> safe;
        vector<Interval> endpoint;
        for (const Real &value : setting.tm_setting.step_end_exp_table) {
            endpoint.push_back(Interval(value));
        }
        const auto setup_end = Clock::now();

        size_t completed_tasks = 0;
        size_t accepted_lane_steps = 0;
        double reach_total = 0.0;
        double postprocess_total = 0.0;

        for (size_t y = 0; y < 4; ++y) {
            for (size_t x = 0; x < 8; ++x) {
                const size_t task_id = y * 8 + x;
                const vector<Interval> box = {
                    Interval(parse_hex(axes.x_lo[x]), parse_hex(axes.x_hi[x])),
                    Interval(parse_hex(axes.y_lo[y]), parse_hex(axes.y_hi[y])),
                };
                Flowpipe initial(box);
                Symbolic_Remainder sr(initial, sr_capacity);
                Result_of_Reachability result;

                const auto reach_start = Clock::now();
                ode.reach(result, initial, requested_steps * h, setting, safe, sr);
                const auto reach_end = Clock::now();
                const size_t accepted = result.flowpipes.size();

                vector<Interval> final_endpoint;
                if (!result.flowpipes.empty()) {
                    result.flowpipes.back().intEvalNormal(
                        final_endpoint, endpoint, order,
                        setting.tm_setting.cutoff_threshold);
                }
                const auto postprocess_end = Clock::now();

                const double reach_s = seconds(reach_start, reach_end);
                const double postprocess_s = seconds(reach_end, postprocess_end);
                reach_total += reach_s;
                postprocess_total += postprocess_s;
                accepted_lane_steps += accepted;
                ++completed_tasks;

                tasks << "{\"plant\":\"" << plant << "\",\"task_id\":"
                      << task_id << ",\"box_hex\":[[\"" << axes.x_lo[x]
                      << "\",\"" << axes.x_hi[x] << "\"],[\""
                      << axes.y_lo[y] << "\",\"" << axes.y_hi[y]
                      << "\"]],\"requested_steps\":" << requested_steps
                      << ",\"accepted_steps\":" << accepted
                      << ",\"native_status\":" << static_cast<int>(result.status)
                      << ",\"reach_seconds\":" << reach_s
                      << ",\"postprocess_seconds\":" << postprocess_s
                      << ",\"final_queue_size\":" << sr.J.size()
                      << ",\"final_endpoint\":[";
                if (final_endpoint.size() == 2) {
                    write_interval(tasks, final_endpoint[0]);
                    tasks << ',';
                    write_interval(tasks, final_endpoint[1]);
                }
                tasks << "]}\n";
            }
        }
        tasks.close();

        const auto summary_start = Clock::now();
        ofstream summary(argv[4]);
        if (!summary) {
            throw runtime_error("summary output unavailable");
        }
        summary << setprecision(17)
                << "{\"plant\":\"" << plant
                << "\",\"mode\":\"native_flowstar_matched_b32\""
                << ",\"state_dimension\":2,\"clock_state\":false"
                << ",\"partition_grid\":[8,4],\"tasks\":32"
                << ",\"requested_steps_per_task\":" << requested_steps
                << ",\"requested_lane_steps\":" << 32 * requested_steps
                << ",\"completed_tasks\":" << completed_tasks
                << ",\"accepted_lane_steps\":" << accepted_lane_steps
                << ",\"h\":" << h << ",\"order\":" << order
                << ",\"cutoff\":1e-10,\"initial_remainder_radius\":1e-4"
                << ",\"symbolic_remainder_capacity\":" << sr_capacity
                << ",\"single_process\":true,\"sequential_tasks\":true"
                << ",\"exact_history_contract_supported\":false"
                << ",\"setup_seconds\":" << seconds(setup_start, setup_end)
                << ",\"reach_seconds\":" << reach_total
                << ",\"postprocess_seconds\":" << postprocess_total
                << ",\"pre_summary_main_seconds\":"
                << seconds(main_start, summary_start) << "}\n";
        summary.close();

        cout << "plant=" << plant << " completed_tasks=" << completed_tasks
             << " accepted_lane_steps=" << accepted_lane_steps
             << " reach_seconds=" << reach_total << '\n';
        return accepted_lane_steps == static_cast<size_t>(32 * requested_steps)
            ? 0
            : 3;
    } catch (const exception &error) {
        cerr << error.what() << '\n';
        return 1;
    }
}
