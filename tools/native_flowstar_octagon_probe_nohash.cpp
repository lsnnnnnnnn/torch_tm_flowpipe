// Isolated, plant-only Flow* probe. Build against the frozen native library;
// do not include this file in any benchmark binary.
#include "Continuous.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flowstar;

static std::vector<Real> direction(unsigned form)
{
    std::vector<Real> d(2, 0);
    d[0] = 1;
    if (form == 1) { d[0] = 0; d[1] = 1; }
    if (form == 2) d[1] = 1;
    if (form == 3) d[1] = -1;
    return d;
}

static Flowpipe propagated_endpoint(const Flowpipe &fp, const Computational_Setting &setting)
{
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

static void observe_new(const Result_of_Reachability &result, size_t &seen,
                        const Computational_Setting &setting, std::ostream &out)
{
    if (result.flowpipes.size() < seen) throw std::runtime_error("flowpipe history shrank");
    const unsigned order = std::max(setting.tm_setting.order, setting.tm_setting.order_max);
    size_t step = 0;
    for (const Flowpipe &fp : result.flowpipes) {
        if (++step <= seen) continue;
        if (fp.domain.empty()) throw std::runtime_error("missing local time domain");
        for (unsigned view = 0; view < 2; ++view) {
            const Flowpipe projected_source = view ? propagated_endpoint(fp, setting) : fp;
            TaylorModelVec<Real> projected;
            projected_source.compose(projected, std::vector<unsigned>{0, 1}, order,
                                     setting.tm_setting.cutoff_threshold);
            if (projected.tms.size() != 2)
                throw std::runtime_error("invalid projected flowpipe");
            for (unsigned form = 0; form < 4; ++form) {
                Interval support;
                projected.rho(support, direction(form), projected_source.domain);
                const double lo = support.inf(), hi = support.sup();
                if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                    throw std::runtime_error("invalid directional interval");
                out << step << ',' << view << ',' << form << ',' << lo << ',' << hi << '\n';
            }
        }
    }
    seen = step;
    out.flush();
    if (!out) throw std::runtime_error("octagon stream write failed");
}

static Result_of_Reachability run(bool observed, Computational_Setting setting,
                                  const ODE<Real> &dynamics, const Flowpipe &initial,
                                  std::ostream &out, double &observation_seconds,
                                  bool single_reach)
{
    Result_of_Reachability result;
    Flowpipe next = initial;
    std::vector<Constraint> safe;
    size_t seen = 0;
    observation_seconds = 0;
    if (single_reach) {
        dynamics.reach(result, next, 0.15, setting, safe);
        if (result.status != COMPLETED_SAFE || result.flowpipes.size() != 3)
            throw std::runtime_error("single harmonic reach did not complete three steps");
        if (observed) {
            const auto start = std::chrono::steady_clock::now();
            observe_new(result, seen, setting, out);
            observation_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
        return result;
    }
    for (unsigned period = 0; period < 3; ++period) {
        dynamics.reach(result, next, 0.05, setting, safe);
        if (result.status != COMPLETED_SAFE || result.flowpipes.size() != period + 1)
            throw std::runtime_error("harmonic solver did not complete expected prefix");
        next = result.fp_end_of_time;
        if (observed) {
            TaylorModelVec<Real> accepted_tm, propagated_tm;
            const Flowpipe rebuilt = propagated_endpoint(result.flowpipes.back(), setting);
            rebuilt.compose(accepted_tm, std::vector<unsigned>{0, 1}, 4,
                            setting.tm_setting.cutoff_threshold);
            next.compose(propagated_tm, std::vector<unsigned>{0, 1}, 4,
                         setting.tm_setting.cutoff_threshold);
            std::vector<Interval> endpoint_pre, endpoint_composed;
            next.tmvPre.intEval(endpoint_pre, next.domain);
            propagated_tm.intEval(endpoint_composed, next.domain);
            const double exact_x_at_left_initial = std::cos((period + 1) * 0.05);
            std::cout << std::setprecision(17) << "endpoint_step=" << (period + 1)
                      << " pre_x=[" << endpoint_pre[0].inf() << ',' << endpoint_pre[0].sup()
                      << "] composed_x=[" << endpoint_composed[0].inf() << ','
                      << endpoint_composed[0].sup() << "] exact_left_x="
                      << exact_x_at_left_initial << '\n';
            for (unsigned form = 0; form < 4; ++form) {
                Interval accepted, propagated;
                accepted_tm.rho(accepted, direction(form), rebuilt.domain);
                propagated_tm.rho(propagated, direction(form), next.domain);
                if (accepted.inf() != propagated.inf() || accepted.sup() != propagated.sup())
                    throw std::runtime_error("fixed-time endpoint differs from propagated endpoint");
            }
        }
        if (observed) {
            const auto start = std::chrono::steady_clock::now();
            observe_new(result, seen, setting, out);
            observation_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
    }
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 3 || (std::string(argv[2]) != "periodic" && std::string(argv[2]) != "single")) {
        std::cerr << "usage: probe output-directory periodic|single\n"; return 2;
    }
    try {
        const bool single_reach = std::string(argv[2]) == "single";
        Variables vars;
        vars.declareVar("x1"); vars.declareVar("x2");
        ODE<Real> dynamics({"x2", "-x1"}, vars);
        Computational_Setting setting(vars);
        setting.setFixedStepsize(0.05, 4);
        setting.setCutoffThreshold(1e-8);
        setting.setRemainderEstimation(std::vector<Interval>(2, Interval(-0.1, 0.1)));
        setting.printOff();
        Flowpipe initial(std::vector<Interval>{Interval(1, 1.2), Interval(0)});
        std::ofstream out(std::string(argv[1]) + "/directional.csv");
        out << std::setprecision(17);
        out << "step,view,form,lo,hi\n";
        double observer_seconds = 0, control_seconds = 0;
        Result_of_Reachability observed = run(true, setting, dynamics, initial, out,
                                             observer_seconds, single_reach);
        Result_of_Reachability control = run(false, setting, dynamics, initial, out,
                                            control_seconds, single_reach);
        auto a = observed.flowpipes.begin(), b = control.flowpipes.begin();
        for (unsigned step = 1; step <= 3; ++step, ++a, ++b) {
            TaylorModelVec<Real> ta, tb;
            a->compose(ta, std::vector<unsigned>{0, 1}, 4, setting.tm_setting.cutoff_threshold);
            b->compose(tb, std::vector<unsigned>{0, 1}, 4, setting.tm_setting.cutoff_threshold);
            std::vector<Interval> raw_pre, composed;
            a->tmvPre.intEval(raw_pre, a->domain);
            ta.intEval(composed, a->domain);
            std::cout << std::setprecision(17) << "step=" << step
                      << " raw_pre_x=[" << raw_pre[0].inf() << ',' << raw_pre[0].sup()
                      << "] composed_x=[" << composed[0].inf() << ',' << composed[0].sup()
                      << "] raw_pre_y=[" << raw_pre[1].inf() << ',' << raw_pre[1].sup()
                      << "] composed_y=[" << composed[1].inf() << ',' << composed[1].sup()
                      << "]\n";
            for (unsigned view = 0; view < 2; ++view) {
                Flowpipe fa = view ? propagated_endpoint(*a, setting) : *a;
                Flowpipe fb = view ? propagated_endpoint(*b, setting) : *b;
                fa.compose(ta, std::vector<unsigned>{0, 1}, 4, setting.tm_setting.cutoff_threshold);
                fb.compose(tb, std::vector<unsigned>{0, 1}, 4, setting.tm_setting.cutoff_threshold);
                for (unsigned form = 0; form < 4; ++form) {
                    Interval ra, rb;
                    ta.rho(ra, direction(form), fa.domain);
                    tb.rho(rb, direction(form), fb.domain);
                    if (ra.inf() != rb.inf() || ra.sup() != rb.sup())
                        throw std::runtime_error("observer changed a directional support");
                }
            }
        }
        Result_of_Reachability reference = observed;
        reference.transformToTaylorModels(setting);
        if (reference.tmv_flowpipes.tmv_flowpipes.size() != 3)
            throw std::runtime_error("native plot conversion lost a segment");
        auto plot_fp = reference.tmv_flowpipes.tmv_flowpipes.begin();
        auto original_fp = observed.flowpipes.begin();
        for (; plot_fp != reference.tmv_flowpipes.tmv_flowpipes.end(); ++plot_fp, ++original_fp) {
            TaylorModelVec<Real> projected;
            original_fp->compose(projected, std::vector<unsigned>{0, 1}, 4,
                                 setting.tm_setting.cutoff_threshold);
            for (unsigned form = 0; form < 4; ++form) {
                Interval current, native;
                projected.rho(current, direction(form), original_fp->domain);
                plot_fp->tmv_flowpipe.rho(native,
                    std::vector<Real>{direction(form)[0], direction(form)[1]}, plot_fp->domain);
                if (current.inf() != native.inf() || current.sup() != native.sup())
                    throw std::runtime_error("direction differs from native plot TM");
            }
        }
        Plot_Setting plot(vars);
        plot.setOutputDims("x1", "x2");
        plot.plot_2D_octagon_MATLAB(std::string(argv[1]) + "/", "native_reference",
                                    reference.tmv_flowpipes, setting);
        std::cout << std::setprecision(17)
                  << "mode=" << argv[2] << " accepted_steps=3 directional_rows=24 reference_plot_segments=3 "
                  << "control_equal=1 observer_seconds=" << observer_seconds << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
