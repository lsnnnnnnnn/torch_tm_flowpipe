#pragma once
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <cstdlib>
namespace arch_ranges {
static void record(const flowstar::Result_of_Reachability& result, uint64_t lane, unsigned physical) {
    const char *path=std::getenv("ARCH_RANGE_LOG"); if (!path) return;
    static std::ofstream out(path,std::ios::binary);
    static std::map<uint64_t,uint64_t> seen;
    if (!out) throw std::runtime_error("range output unavailable");
    uint64_t step=0;
    for (const auto &fp:result.flowpipes) {
        ++step; if(step<=seen[lane])continue;
        auto domain=fp.domain;auto pre=fp.tmvPre;
        double h=domain[0].sup();std::vector<flowstar::Interval> tube,endpoint;
        pre.intEval(tube,domain);domain[0]=flowstar::Interval(h);pre.intEval(endpoint,domain);
        out.write(reinterpret_cast<const char*>(&lane),8);out.write(reinterpret_cast<const char*>(&step),8);out.write(reinterpret_cast<const char*>(&h),8);
        for(unsigned i=0;i<physical;++i){double x[4]={tube[i].inf(),tube[i].sup(),endpoint[i].inf(),endpoint[i].sup()};out.write(reinterpret_cast<const char*>(x),32);}
    }
    if(step<seen[lane])throw std::runtime_error("history replaced");seen[lane]=step;out.flush();if(!out)throw std::runtime_error("range output failed");
}
static void record(const std::vector<flowstar::Result_of_Reachability>& results,unsigned physical){for(uint64_t b=0;b<results.size();++b)record(results[b],b,physical);}
static void boxes(const std::vector<std::vector<flowstar::Interval>>& boxes,const char *path){
    std::ofstream out(path);out<<std::setprecision(17)<<"[";
    for(size_t b=0;b<boxes.size();++b){if(b)out<<",";out<<"[";for(size_t j=0;j<boxes[b].size();++j){if(j)out<<",";out<<"["<<boxes[b][j].inf()<<","<<boxes[b][j].sup()<<"]";}out<<"]";}out<<"]\n";
    if(!out)throw std::runtime_error("boxes output failed");
}
}
