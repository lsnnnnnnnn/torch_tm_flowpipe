#include "Continuous.h"
#include <json/json.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace flowstar;

static std::string hex(double value)
{
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite native value");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hexfloat << value;
    return out.str();
}

static std::string hex(const Real &value)
{
    const double down = value.getValue_RNDD(), up = value.getValue_RNDU();
    if (down != up) throw std::runtime_error("MPFR Real is not exactly binary64");
    return hex(down);
}

static Json::Value endpoints(const Interval &value)
{
    Real lo, hi;
    value.inf(lo);
    value.sup(hi);
    Json::Value result(Json::arrayValue);
    result.append(hex(lo));
    result.append(hex(hi));
    return result;
}

static Json::Value model(const TaylorModel<Real> &tm)
{
    Json::Value result(Json::objectValue), terms(Json::arrayValue);
    for (const auto &term : tm.expansion.terms) {
        Json::Value item(Json::objectValue), degrees(Json::arrayValue);
        item["coefficient"] = hex(term.adoptionCoefficient());
        for (unsigned degree : term.adoptionDegrees()) degrees.append(degree);
        item["degrees"] = degrees;
        terms.append(item);
    }
    result["terms"] = terms;
    result["remainder"] = endpoints(tm.remainder);
    return result;
}

static Json::Value read_json(const char *path)
{
    if (!path) throw std::runtime_error("missing input path");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open input JSON");
    Json::Value result;
    input >> result;
    return result;
}

int main()
{
    try {
        // The frozen copy uses 53-bit MPFR Real/Interval values.
        if (intervalNumPrecision != 53) throw std::runtime_error("unexpected MPFR precision");
        const Json::Value saved = read_json(std::getenv("QUAD_OLD_RPC"));
        const Json::Value correction = read_json(std::getenv("QUAD_CORRECTED_BIASES"));
        const char *output_path = std::getenv("QUAD_TRACE_OUT");
        if (!output_path) throw std::runtime_error("missing output path");
        if (saved["params"]["input_lb"].size() != 12288 ||
            saved["params"]["input_ub"].size() != 12288 ||
            saved["coefficients"]["T"].size() != 1024 ||
            correction["u_min"].size() != 1024 || correction["u_max"].size() != 1024)
            throw std::runtime_error("saved first-batch shape mismatch");

        // Exact lane-0 intervals from the frozen 8x8x8x2x1x1 partition.
        std::vector<Interval> box(16, Interval(0));
        for (unsigned i = 0; i < 3; ++i) box[i] = Interval(-0.4, -0.3);
        box[3] = Interval(-0.4, 0.0);
        box[4] = Interval(-0.4, 0.4);
        box[5] = Interval(-0.4, 0.4);
        Flowpipe initial(box);
        Json::Value trace(Json::objectValue), source(Json::arrayValue), domain(Json::arrayValue);
        Json::Value inputs(Json::arrayValue), rpc_inputs(Json::arrayValue), input_models(Json::arrayValue);
        trace["lane"] = 0;
        trace["mpfr_precision_bits"] = static_cast<int>(intervalNumPrecision);
        for (const Interval &item : box) source.append(endpoints(item));
        for (const Interval &item : initial.domain) domain.append(endpoints(item));
        for (unsigned i = 0; i < 12; ++i) {
            Interval native_input;
            initial.tmvPre.tms[i].intEval(native_input, initial.domain);
            const double lo = native_input.inf(), hi = native_input.sup();
            if (lo != saved["params"]["input_lb"][i].asDouble() ||
                hi != saved["params"]["input_ub"][i].asDouble())
                throw std::runtime_error("lane-0 RPC input mismatch");
            inputs.append(endpoints(native_input));
            Json::Value pair(Json::arrayValue);
            pair.append(hex(saved["params"]["input_lb"][i].asDouble()));
            pair.append(hex(saved["params"]["input_ub"][i].asDouble()));
            rpc_inputs.append(pair);
            input_models.append(model(initial.tmvPre.tms[i]));
        }
        trace["source_box"] = source;
        trace["domain"] = domain;
        trace["native_rpc_input"] = inputs;
        trace["saved_rpc_input"] = rpc_inputs;
        trace["input_tms"] = input_models;

        Matrix<Real> slopes(3, 12, Real(0));
        std::vector<Real> centers;
        std::vector<double> radii;
        for (unsigned j = 0; j < 3; ++j) {
            if (saved["coefficients"]["T"][0][j].size() != 12)
                throw std::runtime_error("slope-row shape mismatch");
            for (unsigned i = 0; i < 12; ++i)
                slopes[j][i] = saved["coefficients"]["T"][0][j][i].asFloat();
            const double lo = correction["u_min"][0][j].asFloat();
            const double hi = correction["u_max"][0][j].asFloat();
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                throw std::runtime_error("invalid corrected residual endpoints");
            centers.push_back((hi + lo) / 2);
            radii.push_back((hi - lo) / 2);
        }
        TaylorModelVec<Real> controls(centers, 17);
        Json::Value outputs(Json::arrayValue);
        const double lower_pad = std::ldexp(1.0, -50);
        for (unsigned j = 0; j < 3; ++j) {
            Json::Value row(Json::objectValue), row_slopes(Json::arrayValue), bias(Json::arrayValue);
            for (unsigned i = 0; i < 12; ++i) {
                controls.tms[j] += initial.tmvPre.tms[i] * slopes[j][i];
                row_slopes.append(hex(slopes[j][i]));
            }
            controls.tms[j].remainder += Interval(-radii[j], radii[j]);
            row["original_tm_remainder"] = endpoints(controls.tms[j].remainder);
            controls.tms[j].remainder += Interval(-lower_pad, 0.0);
            row["expanded_tm_remainder"] = endpoints(controls.tms[j].remainder);
            row["lower_pad"] = hex(lower_pad);
            bias.append(hex(correction["u_min"][0][j].asFloat()));
            bias.append(hex(correction["u_max"][0][j].asFloat()));
            row["output"] = j + 1;
            row["slope"] = row_slopes;
            row["corrected_bias"] = bias;
            row["center"] = hex(centers[j]);
            row["radius"] = hex(radii[j]);
            row["tm"] = model(controls.tms[j]);
            outputs.append(row);
        }
        trace["outputs"] = outputs;
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open trace output");
        output << trace << '\n';
        if (!output) throw std::runtime_error("cannot write trace output");
        std::cout << "LANE0_LOWER_REMAINDER_PAD_TRACE_ONLY_NO_CROWN_NO_ODE\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FIRST_REFUSAL_TRACE " << error.what() << '\n';
        return 3;
    }
}
