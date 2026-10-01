#pragma once
#include <cmath>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <stdexcept>
#include <cstdlib>

namespace author_matched {
using namespace flowstar;

// Explicit comparison contract: the same binary64 boxes as the GPU driver.
// With no environment option the original driver's initial sets stay intact.
static void initial(std::vector<Flowpipe> & sets) {
    const char * path=std::getenv("AUTHOR_INITIAL_BOXES");
    if (!path) return;
    std::ifstream stream(path);
    Json::Value boxes;
    if (!(stream >> boxes) || boxes.size()!=sets.size())
        throw std::runtime_error("invalid shared initial boxes");
    for (size_t lane=0;lane<sets.size();++lane) {
        if (boxes[(int)lane].size()!=sets[lane].tmvPre.tms.size())
            throw std::runtime_error("shared initial dimension mismatch");
        std::vector<Interval> box;
        for (const auto & pair: boxes[(int)lane]) {
            if (pair.size()!=2 || !std::isfinite(pair[0].asDouble()) ||
                !std::isfinite(pair[1].asDouble()) || pair[0].asDouble()>pair[1].asDouble())
                throw std::runtime_error("invalid shared interval");
            box.emplace_back(pair[0].asDouble(),pair[1].asDouble());
        }
        sets[lane]=Flowpipe(box);
    }
}

static void reach(const ODE<Real> & ode, Result_of_Reachability & result,
        const Flowpipe & initialSet, double duration, Computational_Setting & setting,
        const std::vector<Constraint> & safeSet, Symbolic_Remainder & sr) {
    if (!std::getenv("AUTHOR_FIXED_STEPS")) {
        ode.reach(result,initialSet,duration,setting,safeSet,sr);
        return;
    }
    const double h=setting.tm_setting.step_exp_table[1].sup();
    const int steps=std::lround(duration/h);
    if (steps<=0 || std::fabs(steps*h-duration)>1e-15 ||
        setting.tm_setting.step_min>0 || setting.tm_setting.order_max>0)
        throw std::runtime_error("fixed-count contract requires a fixed divisible step/order");
    const std::vector<Constraint> invariant;
    Flowpipe current=initialSet,next;
    int status=COMPLETED_SAFE;
    // Same advance, safety, SR reset and endpoint construction as Continuous.h:
    // 832-911 / 1148-1205. Only count and unchanged full local domain differ.
    for (int k=0;k<steps;++k) {
        const int accepted=current.advance(next,ode.expressions,setting.tm_setting,invariant,setting.g_setting,sr);
        if (accepted!=1) {
            status=status==COMPLETED_SAFE ? UNCOMPLETED_SAFE :
                   status==COMPLETED_UNSAFE ? UNCOMPLETED_UNSAFE : UNCOMPLETED_UNKNOWN;
            break;
        }
        if (next.domain[0].inf()!=0.0 || next.domain[0].sup()!=h)
            throw std::runtime_error("advance returned unexpected time domain");
        next.safety=safeSet.empty() ? SAFE : next.safetyChecking(safeSet,setting.tm_setting,setting.g_setting);
        result.flowpipes.push_back(next);
        if (next.safety==UNSAFE) {status=COMPLETED_UNSAFE;break;}
        if (next.safety==UNKNOWN && status==COMPLETED_SAFE) status=COMPLETED_UNKNOWN;
        current=next;
        if (sr.J.size()>=sr.max_size) sr.reset(current.tmvPre.tms.size());
    }
    result.status=status;
    if (!result.flowpipes.empty()) {
        Flowpipe fp=result.flowpipes.back();
        result.fp_end_of_time=fp;
        Real t;
        fp.domain[0].sup(t);
        std::vector<Real> powers;
        powers.push_back(1);
        powers.push_back(t);
        Real tmp=t;
        for (unsigned int i=2;i<=setting.tm_setting.step_end_exp_table.size();++i) {
            tmp*=t;
            powers.push_back(tmp);
        }
        fp.tmvPre.evaluate_time(result.fp_end_of_time.tmvPre,powers);
    }
}
}
