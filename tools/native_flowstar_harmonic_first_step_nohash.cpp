// Diagnostic only: inspect frozen Flow* first accepted harmonic step.
#include "Continuous.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace flowstar;

static void show(const char *label, const TaylorModel<Real> &tm)
{
    std::cout << label << " remainder=[" << tm.remainder.inf() << ','
              << tm.remainder.sup() << "] terms=" << tm.expansion.terms.size() << '\n';
    for (const auto &term : tm.expansion.terms) {
        std::cout << "  coefficient=" << term.adoptionCoefficient().toDouble() << " degrees=";
        for (unsigned degree : term.adoptionDegrees()) std::cout << degree << ',';
        std::cout << '\n';
    }
}

int main()
{
    Variables vars;
    vars.declareVar("x1"); vars.declareVar("x2");
    ODE<Real> dynamics({"x2", "-x1"}, vars);
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.05, 4);
    setting.setCutoffThreshold(1e-8);
    setting.setRemainderEstimation(std::vector<Interval>(2, Interval(-0.1, 0.1)));
    setting.printOff();
    Flowpipe initial(std::vector<Interval>{Interval(1, 1.2), Interval(0)});
    Result_of_Reachability result;
    std::vector<Constraint> safe;
    dynamics.reach(result, initial, 0.05, setting, safe);
    if (result.status != COMPLETED_SAFE || result.flowpipes.size() != 1) return 2;
    const Flowpipe &fp = result.flowpipes.front();
    const Flowpipe &end = result.fp_end_of_time;
    std::cout << std::setprecision(17);
    show("initial.tmvPre.x", initial.tmvPre.tms[0]);
    show("initial.tmv.x", initial.tmv.tms[0]);
    show("accepted.tmvPre.x", fp.tmvPre.tms[0]);
    show("accepted.tmvPre.y", fp.tmvPre.tms[1]);
    show("accepted.tmv.x", fp.tmv.tms[0]);
    show("endpoint.tmvPre.x", end.tmvPre.tms[0]);
    show("endpoint.tmvPre.y", end.tmvPre.tms[1]);
    show("endpoint.tmv.x", end.tmv.tms[0]);
    TaylorModel<Interval> rhs;
    std::list<Interval> intermediate;
    dynamics.expressions[0].evaluate(rhs, fp.tmvPre.tms, 3,
        setting.tm_setting.step_exp_table, setting.tm_setting.cutoff_threshold,
        3, intermediate, setting.g_setting);
    std::cout << "validated_rhs_xdot_remainder=[" << rhs.remainder.inf() << ','
              << rhs.remainder.sup() << "] terms=" << rhs.expansion.terms.size() << '\n';
    for (const auto &term : rhs.expansion.terms) {
        std::cout << "  coefficient=[" << term.adoptionCoefficient().inf() << ','
                  << term.adoptionCoefficient().sup() << "] degrees=";
        for (unsigned degree : term.adoptionDegrees()) std::cout << degree << ',';
        std::cout << '\n';
    }
    std::vector<Interval> endpoint_pre, endpoint_composed;
    end.tmvPre.intEval(endpoint_pre, end.domain);
    end.intEval(endpoint_composed, 4, setting.tm_setting.cutoff_threshold);
    std::cout << "endpoint_pre_x=[" << endpoint_pre[0].inf() << ',' << endpoint_pre[0].sup()
              << "] endpoint_composed_x=[" << endpoint_composed[0].inf() << ','
              << endpoint_composed[0].sup() << "] cos(0.05)=" << std::cos(0.05) << '\n';
    return 0;
}
