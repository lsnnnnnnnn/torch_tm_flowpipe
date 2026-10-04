#include "Continuous.h"
#include <json/json.h>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
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

static double exact_hex(const Json::Value &value)
{
    if (!value.isString()) throw std::runtime_error("endpoint is not a hex string");
    const std::string input = value.asString();
    char *end = nullptr;
    errno = 0;
    const double result = std::strtod(input.c_str(), &end);
    if (errno || !end || *end || !std::isfinite(result) || input.find("0x") == std::string::npos)
        throw std::runtime_error("invalid finite hex endpoint");
    return result;
}

int main()
{
    unsigned lane = 0, output_index = 0;
    try {
        // The frozen copy uses 53-bit MPFR Real/Interval values.
        if (intervalNumPrecision != 53) throw std::runtime_error("unexpected MPFR precision");
        const Json::Value saved = read_json(std::getenv("QUAD_OLD_RPC"));
        const Json::Value correction = read_json(std::getenv("QUAD_CORRECTED_BIASES"));
        const Json::Value plan = read_json(std::getenv("QUAD_REMAINDER_PLAN"));
        const char *prior_path = std::getenv("QUAD_PRIOR_TRACE");
        if (!prior_path) throw std::runtime_error("missing prior trace path");
        std::ifstream prior_trace(prior_path);
        if (!prior_trace || plan["rows"].size() != 1024)
            throw std::runtime_error("prior trace or plan shape mismatch");
        const char *output_path = std::getenv("QUAD_TRACE_OUT");
        if (!output_path) throw std::runtime_error("missing output path");
        if (saved["params"]["input_lb"].size() != 12288 ||
            saved["params"]["input_ub"].size() != 12288 ||
            saved["coefficients"]["T"].size() != 1024 ||
            correction["u_min"].size() != 1024 || correction["u_max"].size() != 1024)
            throw std::runtime_error("saved first-batch shape mismatch");

        // Reproduce the frozen source partition and its loop order exactly.
        Interval x1(-0.4, 0.4), x2(-0.4, 0.4), x3(-0.4, 0.4);
        Interval x4(-0.4, 0.4), x5(-0.4, 0.4), x6(-0.4, 0.4);
        std::list<Interval> a, b, c, d, e, f;
        x1.split(a, 8); x2.split(b, 8); x3.split(c, 8);
        x4.split(d, 2); x5.split(e, 1); x6.split(f, 1);
        std::vector<std::vector<Interval>> boxes;
        boxes.reserve(1024);
        for (const Interval &v1 : a) for (const Interval &v2 : b)
        for (const Interval &v3 : c) for (const Interval &v4 : d)
        for (const Interval &v5 : e) for (const Interval &v6 : f) {
            std::vector<Interval> box(16, Interval(0));
            box[0] = v1; box[1] = v2; box[2] = v3;
            box[3] = v4; box[4] = v5; box[5] = v6;
            boxes.push_back(box);
        }
        if (boxes.size() != 1024) throw std::runtime_error("source partition size mismatch");
        std::ifstream prior(output_path);
        if (prior.good()) throw std::runtime_error("refusing existing trace path");
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open trace output");
        Json::FastWriter writer;
        for (lane = 0; lane < boxes.size(); ++lane) {
            output_index = 0;
            std::string prior_line;
            if (!std::getline(prior_trace, prior_line)) throw std::runtime_error("missing prior trace row");
            Json::Value expected;
            std::istringstream prior_row(prior_line);
            prior_row >> expected;
            if (expected["lane"].asUInt() != lane || plan["rows"][lane]["lane"].asUInt() != lane ||
                expected["outputs"].size() != 3 || plan["rows"][lane]["outputs"].size() != 3)
                throw std::runtime_error("prior or plan lane/output mismatch");
            const std::vector<Interval> &box = boxes[lane];
            Flowpipe initial(box);
            Json::Value trace(Json::objectValue), source(Json::arrayValue), domain(Json::arrayValue);
            Json::Value inputs(Json::arrayValue), rpc_inputs(Json::arrayValue), input_models(Json::arrayValue);
            trace["lane"] = lane;
            trace["mpfr_precision_bits"] = static_cast<int>(intervalNumPrecision);
            for (const Interval &item : box) source.append(endpoints(item));
            for (const Interval &item : initial.domain) domain.append(endpoints(item));
            for (unsigned i = 0; i < 12; ++i) {
                const unsigned pos = lane * 12 + i;
                Interval native_input;
                initial.tmvPre.tms[i].intEval(native_input, initial.domain);
                const double lo = native_input.inf(), hi = native_input.sup();
                if (lo != saved["params"]["input_lb"][pos].asDouble() ||
                    hi != saved["params"]["input_ub"][pos].asDouble())
                    throw std::runtime_error("RPC input mismatch at coordinate " + std::to_string(i + 1));
                inputs.append(endpoints(native_input));
                Json::Value pair(Json::arrayValue);
                pair.append(hex(saved["params"]["input_lb"][pos].asDouble()));
                pair.append(hex(saved["params"]["input_ub"][pos].asDouble()));
                rpc_inputs.append(pair);
                input_models.append(model(initial.tmvPre.tms[i]));
            }
            trace["source_box"] = source;
            trace["domain"] = domain;
            trace["native_rpc_input"] = inputs;
            trace["saved_rpc_input"] = rpc_inputs;
            trace["input_tms"] = input_models;
            Json::Value expected_inputs = expected;
            expected_inputs.removeMember("outputs");
            if (writer.write(trace) != writer.write(expected_inputs)) throw std::runtime_error("input models/source/RPC/domain changed");

            Matrix<Real> slopes(3, 12, Real(0));
            std::vector<Real> centers;
            std::vector<double> radii;
            for (unsigned j = 0; j < 3; ++j) {
                if (saved["coefficients"]["T"][lane][j].size() != 12)
                    throw std::runtime_error("slope-row shape mismatch");
                for (unsigned i = 0; i < 12; ++i)
                    slopes[j][i] = saved["coefficients"]["T"][lane][j][i].asFloat();
                const double lo = correction["u_min"][lane][j].asFloat();
                const double hi = correction["u_max"][lane][j].asFloat();
                if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                    throw std::runtime_error("invalid corrected residual endpoints");
                centers.push_back((hi + lo) / 2);
                radii.push_back((hi - lo) / 2);
            }
            TaylorModelVec<Real> controls(centers, 17);
            Json::Value outputs(Json::arrayValue);
            for (unsigned j = 0; j < 3; ++j) {
                output_index = j + 1;
                Json::Value row(Json::objectValue), row_slopes(Json::arrayValue), bias(Json::arrayValue);
                for (unsigned i = 0; i < 12; ++i) {
                    controls.tms[j] += initial.tmvPre.tms[i] * slopes[j][i];
                    row_slopes.append(hex(slopes[j][i]));
                }
                controls.tms[j].remainder += Interval(-radii[j], radii[j]);
                row["original_tm_remainder"] = endpoints(controls.tms[j].remainder);
                bias.append(hex(correction["u_min"][lane][j].asFloat()));
                bias.append(hex(correction["u_max"][lane][j].asFloat()));
                row["output"] = j + 1;
                row["slope"] = row_slopes;
                row["corrected_bias"] = bias;
                row["center"] = hex(centers[j]);
                row["radius"] = hex(radii[j]);
                row["tm"] = model(controls.tms[j]);
                Json::Value expected_output = expected["outputs"][j];
                expected_output.removeMember("expanded_tm_remainder");
                expected_output.removeMember("lower_pad");
                expected_output["tm"]["remainder"] = expected_output["original_tm_remainder"];
                if (writer.write(row) != writer.write(expected_output)) throw std::runtime_error("original output construction changed");
                const Json::Value &planned = plan["rows"][lane]["outputs"][j];
                if (planned["output"].asUInt() != j + 1 || planned["remainder"].size() != 2)
                    throw std::runtime_error("planned output shape mismatch");
                const double lo = exact_hex(planned["remainder"][0]);
                const double hi = exact_hex(planned["remainder"][1]);
                if (lo > controls.tms[j].remainder.inf() || hi < controls.tms[j].remainder.sup() || lo > hi)
                    throw std::runtime_error("planned remainder does not preserve original");
                controls.tms[j].remainder = Interval(lo, hi);
                row["replacement_tm_remainder"] = endpoints(controls.tms[j].remainder);
                if (exact_hex(row["replacement_tm_remainder"][0]) != lo ||
                    exact_hex(row["replacement_tm_remainder"][1]) != hi)
                    throw std::runtime_error("native endpoint replacement differs from plan");
                row["tm"] = model(controls.tms[j]);
                outputs.append(row);
            }
            trace["outputs"] = outputs;
            output << writer.write(trace);
            output.flush();
            if (!output) throw std::runtime_error("cannot write trace output");
        }
        std::string extra;
        if (std::getline(prior_trace, extra)) throw std::runtime_error("extra prior trace row");
        std::cout << "ALLBOX_ADAPTIVE_REMAINDER_REPLACEMENT_TRACE_ONLY_NO_CROWN_NO_ODE\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FIRST_REFUSAL_TRACE lane=" << lane << " output=" << output_index << ' ' << error.what() << '\n';
        return 3;
    }
}
