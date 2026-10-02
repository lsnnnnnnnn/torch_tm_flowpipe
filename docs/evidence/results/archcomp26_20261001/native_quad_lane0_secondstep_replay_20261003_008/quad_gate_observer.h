#pragma once

// Diagnostic only: saved first-call RPC replay, lane zero, at most two small steps.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <jsoncpp/json/json.h>

namespace quad_gate {
using namespace flowstar;

static const char *directory() {
    const char *value = std::getenv("QUAD_GATE_OUTPUT_DIR");
    if (!value || !*value) throw std::runtime_error("QUAD_GATE_OUTPUT_DIR is required");
    return value;
}

static void save_rpc(const Json::Value &params, const Json::Value &coefficients) {
    Json::Value record;
    record["params"] = params;
    record["coefficients"] = coefficients;
    std::ofstream out(std::string(directory()) + "/rpc.json");
    if (!out) throw std::runtime_error("cannot open RPC receipt");
    out << record << '\n';
    if (!out) throw std::runtime_error("cannot write RPC receipt");
}

static std::vector<Real> direction(unsigned form) {
    std::vector<Real> d(2, 0);
    d[0] = 1;
    if (form == 1) { d[0] = 0; d[1] = 1; }
    if (form == 2) d[1] = 1;
    if (form == 3) d[1] = -1;
    return d;
}

static Flowpipe propagated_endpoint(const Flowpipe &fp,
                                     const Computational_Setting &setting) {
    Flowpipe end = fp;
    Real h;
    fp.domain[0].sup(h);
    std::vector<Real> powers{1, h};
    Real power = h;
    for (unsigned i = 2; i <= setting.tm_setting.step_end_exp_table.size(); ++i) {
        power *= h;
        powers.push_back(power);
    }
    fp.tmvPre.evaluate_time(end.tmvPre, powers);
    return end;
}

static void save_state(unsigned lane, const Result_of_Reachability &result,
                       const Computational_Setting &setting, double seconds) {
    const std::ios::openmode mode = lane ? std::ios::app : std::ios::trunc;
    std::ofstream state(std::string(directory()) + "/state.csv", mode);
    if (!state) throw std::runtime_error("cannot open state receipt");
    if (!lane) state << "lane,status,accepted_steps,seconds\n";
    state << std::setprecision(17) << lane << ',' << result.status << ','
          << result.flowpipes.size() << ',' << seconds << '\n';
    if (!state) throw std::runtime_error("cannot write state receipt");

    std::ofstream ranges(std::string(directory()) + "/ranges.bin",
                         std::ios::binary | mode);
    if (!ranges) throw std::runtime_error("cannot open range receipt");
    std::uint64_t step = 0, lane64 = lane;
    for (const Flowpipe &fp : result.flowpipes) {
        ++step;
        auto domain = fp.domain;
        auto pre = fp.tmvPre;
        const double duration = domain[0].sup();
        std::vector<Interval> tube, endpoint;
        pre.intEval(tube, domain);
        domain[0] = Interval(duration);
        pre.intEval(endpoint, domain);
        if (tube.size() < 12 || endpoint.size() < 12 ||
            !std::isfinite(duration) || duration <= 0)
            throw std::runtime_error("invalid native range record");
        ranges.write(reinterpret_cast<const char *>(&lane64), sizeof(lane64));
        ranges.write(reinterpret_cast<const char *>(&step), sizeof(step));
        ranges.write(reinterpret_cast<const char *>(&duration), sizeof(duration));
        for (unsigned coord = 0; coord < 12; ++coord) {
            double row[4] = {tube[coord].inf(), tube[coord].sup(),
                             endpoint[coord].inf(), endpoint[coord].sup()};
            if (!std::isfinite(row[0]) || !std::isfinite(row[1]) ||
                !std::isfinite(row[2]) || !std::isfinite(row[3]) ||
                row[0] > row[1] || row[2] > row[3])
                throw std::runtime_error("nonfinite or reversed native range");
            ranges.write(reinterpret_cast<const char *>(row), sizeof(row));
        }
    }
    if (!ranges) throw std::runtime_error("cannot write range receipt");
    if (result.flowpipes.empty() ||
        (result.status != COMPLETED_SAFE && result.status != COMPLETED_UNSAFE &&
         result.status != COMPLETED_UNKNOWN)) return;

    std::ofstream axes(std::string(directory()) + "/terminal_axes.csv", mode);
    if (!axes) throw std::runtime_error("cannot open terminal axes");
    axes << std::setprecision(17);
    if (!lane) axes << "lane,coord,pre_lo,pre_hi,composed_lo,composed_hi\n";
    const Flowpipe &end = result.fp_end_of_time;
    std::vector<Interval> pre_axes, composed_axes;
    end.tmvPre.intEval(pre_axes, end.domain);
    TaylorModelVec<Real> composed;
    std::vector<unsigned> first_twelve;
    for (unsigned i = 0; i < 12; ++i) first_twelve.push_back(i);
    const unsigned order = std::max(setting.tm_setting.order, setting.tm_setting.order_max);
    end.compose(composed, first_twelve, order, setting.tm_setting.cutoff_threshold);
    composed.intEval(composed_axes, end.domain);
    if (pre_axes.size() < 12 || composed_axes.size() != 12)
        throw std::runtime_error("missing terminal axes");
    for (unsigned i = 0; i < 12; ++i)
        axes << lane << ',' << (i + 1) << ',' << pre_axes[i].inf() << ',' << pre_axes[i].sup()
             << ',' << composed_axes[i].inf() << ',' << composed_axes[i].sup() << '\n';

    if (!std::getenv("QUAD_GATE_OBSERVER")) return;
    std::ofstream oct(std::string(directory()) + "/octagon.csv", mode);
    if (!oct) throw std::runtime_error("cannot open octagon receipt");
    oct << std::setprecision(17);
    if (!lane) oct << "lane,step,view,form,lo,hi\n";
    unsigned observer_step = 0;
    for (const Flowpipe &fp : result.flowpipes) {
        ++observer_step;
        for (unsigned view = 0; view < 2; ++view) {
            Flowpipe source = view ? propagated_endpoint(fp, setting) : fp;
            TaylorModelVec<Real> projected;
            source.compose(projected, std::vector<unsigned>{0, 1}, order,
                           setting.tm_setting.cutoff_threshold);
            for (unsigned form = 0; form < 4; ++form) {
                Interval range;
                projected.rho(range, direction(form), source.domain);
                const double lo = range.inf(), hi = range.sup();
                if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                    throw std::runtime_error("nonfinite or reversed octagon interval");
                oct << lane << ',' << observer_step << ',' << view << ','
                    << form << ',' << lo << ',' << hi << '\n';
            }
        }
    }
    // For the sole step, the propagated endpoint must be the solver's own end.
    Flowpipe rebuilt = propagated_endpoint(result.flowpipes.back(), setting);
    TaylorModelVec<Real> rebuilt_xy, final_xy;
    rebuilt.compose(rebuilt_xy, std::vector<unsigned>{0, 1}, order,
                    setting.tm_setting.cutoff_threshold);
    end.compose(final_xy, std::vector<unsigned>{0, 1}, order,
                setting.tm_setting.cutoff_threshold);
    for (unsigned form = 0; form < 4; ++form) {
        Interval a, b;
        rebuilt_xy.rho(a, direction(form), rebuilt.domain);
        final_xy.rho(b, direction(form), end.domain);
        if (a.inf() != b.inf() || a.sup() != b.sup())
            throw std::runtime_error("propagated endpoint differs from solver endpoint");
    }
}
}
