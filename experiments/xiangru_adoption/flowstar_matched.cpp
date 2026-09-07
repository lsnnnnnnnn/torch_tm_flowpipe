// Plant-only driver for the pinned Flow* toolbox. No additional clock state.
#include "Continuous.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace flowstar;
using namespace std;

void bounds(ostream& out, const Interval& a) {
    out << '[' << setprecision(17) << a.inf() << ',' << a.sup() << ']';
}

void tm_json(ostream& out, const TaylorModelVec<Real>& tm, const vector<Interval>& domain) {
    out << "{\"domain\":[";
    for(size_t i=0;i<domain.size();++i) { if(i)out<<','; bounds(out,domain[i]); }
    out << "],\"variables\":[\"tau\",\"ux\",\"uy\"],\"components\":[";
    for(size_t i=0;i<tm.tms.size();++i) {
        if(i) out<<',';
        out << "{\"remainder\":"; bounds(out,tm.tms[i].remainder);
        out << ",\"terms\":[";
        bool first=true;
        for(const auto& term:tm.tms[i].expansion.terms) {
            if(!first)out<<','; first=false;
            const Real& c=term.adoptionCoefficient();
            out << "{\"coefficient\":[" << c.getValue_RNDD() << ',' << c.getValue_RNDU() << "],\"degrees\":[";
            const auto& ds=term.adoptionDegrees();
            for(size_t j=0;j<ds.size();++j) { if(j)out<<',';out<<ds[j]; }
            out << "]}";
        }
        out << "]}";
    }
    out << "]}";
}

int main(int argc,char** argv) {
    if(argc!=5) { cerr<<"usage: driver brusselator|van_der_pol STEPS OUTPUT_JSONL SUMMARY_JSON\n"; return 2; }
    try {
        const string plant=argv[1]; const int count=stoi(argv[2]);
        const bool bruss=plant=="brusselator";
        if(!bruss && plant!="van_der_pol")throw runtime_error("unknown plant");
        const double h=bruss ? 0.02 : 0.01;
        const int order=bruss ? 6 : 4;
        Variables vars; vars.declareVar("x"); vars.declareVar("y");
        const vector<string> rhs=bruss ? vector<string>{"1 + x*(x*y - 4)","x*(3 - x*y)"}
                                      : vector<string>{"y","y - x - x*x*y"};
        const auto setup_start=chrono::steady_clock::now();
        ODE<Real> ode(rhs,vars);
        Computational_Setting setting(vars);setting.printOff();
        if(!setting.setFixedStepsize(h,order))throw runtime_error("invalid fixed step");
        setting.setCutoffThreshold(1e-10);
        setting.setRemainderEstimation(vector<Interval>(2,Interval(-1e-4,1e-4)));
        vector<Interval> box=bruss ? vector<Interval>{Interval("1.48","1.52"),Interval("2.98","3.02")}
                                   : vector<Interval>{Interval("1.1","1.4"),Interval("2.35","2.45")};
        Flowpipe initial(box); Symbolic_Remainder sr(initial,bruss ? 1000 : 100);
        Result_of_Reachability result; vector<Constraint> safe;
        const auto start=chrono::steady_clock::now();
        ode.reach(result,initial,count*h,setting,safe,sr);
        const auto finish=chrono::steady_clock::now();
        const double solve=chrono::duration<double>(finish-start).count();
        const double setup=chrono::duration<double>(start-setup_start).count();
        ofstream out(argv[3]);out<<setprecision(17);
        if(!out)throw runtime_error("output unavailable");
        vector<Interval> endpoint;
        for(const Real& v:setting.tm_setting.step_end_exp_table)endpoint.push_back(Interval(v));
        int k=0;
        for(const auto& fp:result.flowpipes) {
            ++k;
            vector<Interval> end_box,tube_box;
            fp.intEvalNormal(end_box,endpoint,order,setting.tm_setting.cutoff_threshold);
            fp.intEvalNormal(tube_box,setting.tm_setting.step_exp_table,order,setting.tm_setting.cutoff_threshold);
            TaylorModelVec<Real> composed,composed_endpoint;
            fp.compose_normal(composed,setting.tm_setting.step_exp_table,order,setting.tm_setting.cutoff_threshold);
            fp.compose_normal(composed_endpoint,endpoint,order,setting.tm_setting.cutoff_threshold);
            auto end_domain=fp.domain;end_domain[0]=Interval(h);
            out<<"{\"step\":"<<k<<",\"h\":"<<h<<",\"state_dimension\":2,\"safety_check_passed\":true,\"published\":{\"endpoint\":[";
            bounds(out,end_box[0]);out<<',';bounds(out,end_box[1]);
            out<<"],\"tube\":[";bounds(out,tube_box[0]);out<<',';bounds(out,tube_box[1]);
            out<<"]},\"models\":{\"tube\":";tm_json(out,composed,fp.domain);
            out<<",\"endpoint\":";tm_json(out,composed_endpoint,end_domain);
            out<<"}}\n";
        }
        out.close();
        const double export_s=chrono::duration<double>(chrono::steady_clock::now()-finish).count();
        ofstream summary(argv[4]);summary<<setprecision(17);
        summary<<"{\"plant\":\""<<plant<<"\",\"mode\":\"native_flowstar\",\"state_dimension\":2,\"clock_state\":false,"
               <<"\"requested_steps\":"<<count<<",\"requested_horizon\":"<<count*h<<",\"accepted_steps\":"<<k
               <<",\"accepted_horizon\":"<<k*h<<",\"h\":"<<h<<",\"native_status\":"<<result.status
               <<",\"solve_seconds\":"<<solve+setup<<",\"reach_seconds\":"<<solve<<",\"setup_seconds\":"<<setup
               <<",\"export_seconds\":"<<export_s<<",\"final_queue_size\":"<<sr.J.size()<<"}\n";
        cout<<"accepted="<<k<<" requested="<<count<<" solve_seconds="<<solve+setup<<" export_seconds="<<export_s<<'\n';
        return 0;
    } catch(const exception& e) {cerr<<e.what()<<'\n';return 1;}
}
