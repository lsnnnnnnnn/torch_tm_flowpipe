#!/usr/bin/env python3
"""Derive a no-RPC, no-ODE control-construction probe from the frozen QUAD copy."""

import difflib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/quad_allbox.cpp"


def change(source, old, new):
    assert source.count(old) == 1, f"expected one source anchor: {old[:60]!r}"
    return source.replace(old, new)


def main():
    old_source = ORIGINAL.read_text()
    source = change(old_source,
        '    // RPC client\n    HttpClient httpclient("http://127.0.0.1:5000");\n    Client cl(httpclient, JSONRPC_CLIENT_V2);',
        '    // Isolated saved-RPC controller construction: no server or RPC connection.')
    source = change(source,
        '        cout << "Calling CROWN." << endl;\n'
        '        cl.CallMethod("CROWN_reach", params, output_coefficients);\n'
        '        quad_gate::save_rpc(params, output_coefficients);',
        r'''        cout << "Loading saved first CROWN batch and corrected residual endpoints." << endl;
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
        control_trace << std::setprecision(17);''')
    source = change(source,
        '                tmv_output.tms[j].remainder += remainder_temp;\n',
        r'''                tmv_output.tms[j].remainder += remainder_temp;

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
''')
    marker = '        // Flow*\n'
    assert source.count(marker) == 1
    source = source.split(marker)[0] + (
        '        cout << "CONTROL_CONSTRUCTION_ACCEPTED_3072_NO_ODE" << endl;\n'
        '        return 0;\n'
        '    }\n'
        '}\n')
    (HERE / "control_gate.cpp").write_text(source)
    diff = difflib.unified_diff(old_source.splitlines(True), source.splitlines(True),
                                fromfile="frozen_quad_allbox.cpp", tofile="control_gate.cpp")
    (HERE / "source.patch").write_text("".join(diff))
    print(f"generated {len(source)} source bytes")


if __name__ == "__main__":
    main()
