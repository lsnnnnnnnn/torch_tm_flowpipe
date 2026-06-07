#include "../../../flowstar/flowstar-toolbox/Continuous.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

const vector<string> kHeaders = {
    "source",
    "mode",
    "step_index",
    "adaptive_attempt_index",
    "t_before",
    "h",
    "t_after",
    "accepted",
    "rejected",
    "status",
    "message",
    "tmv_pre_range_width_x",
    "tmv_pre_range_width_y",
    "tmv_pre_range_width_sum",
    "tmv_right_range_width_x",
    "tmv_right_range_width_y",
    "tmv_right_range_width_sum",
    "tmv_right_normal_range_width_x",
    "tmv_right_normal_range_width_y",
    "tmv_right_normal_range_width_sum",
    "endpoint_pre_center_width_x",
    "endpoint_pre_center_width_y",
    "endpoint_pre_center_width_sum",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "inv_scale_x",
    "inv_scale_y",
    "new_x0_width_x",
    "new_x0_width_y",
    "new_x0_width_sum",
    "target_remainder_width_x",
    "target_remainder_width_y",
    "target_remainder_width_sum",
    "picard_no_remainder_residual_width_x",
    "picard_no_remainder_residual_width_y",
    "picard_no_remainder_residual_width_sum",
    "picard_ctrunc_normal_residual_width_x",
    "picard_ctrunc_normal_residual_width_y",
    "picard_ctrunc_normal_residual_width_sum",
    "cutoff_polynomial_difference_width_x",
    "cutoff_polynomial_difference_width_y",
    "cutoff_polynomial_difference_width_sum",
    "symbolic_J_size",
    "symbolic_Phi_L_size",
    "scalar_x",
    "scalar_y",
    "symbolic_J_width_x",
    "symbolic_J_width_y",
    "symbolic_J_width_sum",
    "symbolic_propagated_width_x",
    "symbolic_propagated_width_y",
    "symbolic_propagated_width_sum",
    "final_flowpipe_width_x",
    "final_flowpipe_width_y",
    "final_flowpipe_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "residual_over_target_x",
    "residual_over_target_y",
    "residual_over_target_sum"};

typedef map<string, string> Row;

double real_to_double(const Real &value)
{
    return value.toDouble();
}

double interval_width(const Interval &interval)
{
    return interval.width();
}

string format_double(double value)
{
    if (!std::isfinite(value))
    {
        return "";
    }
    ostringstream out;
    out << setprecision(17) << value;
    return out.str();
}

string format_size(size_t value)
{
    ostringstream out;
    out << value;
    return out.str();
}

void set_value(Row &row, const string &key, const string &value)
{
    row[key] = value;
}

void set_value(Row &row, const string &key, const char *value)
{
    row[key] = value == NULL ? "" : string(value);
}

void set_value(Row &row, const string &key, double value)
{
    row[key] = format_double(value);
}

void set_value(Row &row, const string &key, bool value)
{
    row[key] = value ? "true" : "false";
}

double width_sum(const vector<Interval> &boxes)
{
    double total = 0.0;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        total += interval_width(boxes[i]);
    }
    return total;
}

void set_widths(Row &row, const string &prefix, const vector<Interval> &boxes)
{
    double total = 0.0;
    if (boxes.size() > 0)
    {
        const double width = interval_width(boxes[0]);
        set_value(row, prefix + "_width_x", width);
        total += width;
    }
    if (boxes.size() > 1)
    {
        const double width = interval_width(boxes[1]);
        set_value(row, prefix + "_width_y", width);
        total += width;
    }
    set_value(row, prefix + "_width_sum", total);
}

vector<Interval> matrix_column_intervals(const Matrix<Interval> &matrix)
{
    vector<Interval> out;
    for (unsigned int i = 0; i < matrix.rows(); ++i)
    {
        out.push_back(matrix.data[i * matrix.cols()]);
    }
    return out;
}

vector<Interval> matrix_abs_range(const Matrix<Real> &matrix)
{
    vector<Interval> rows;
    for (unsigned int i = 0; i < matrix.rows(); ++i)
    {
        double radius = 0.0;
        for (unsigned int j = 0; j < matrix.cols(); ++j)
        {
            radius += std::fabs(real_to_double(matrix.data[i * matrix.cols() + j]));
        }
        rows.push_back(Interval(-radius, radius));
    }
    return rows;
}

string csv_escape(const string &value)
{
    bool needs_quote = false;
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == ',' || value[i] == '"' || value[i] == '\n' || value[i] == '\r')
        {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote)
    {
        return value;
    }
    string escaped = "\"";
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += value[i];
        }
    }
    escaped += "\"";
    return escaped;
}

void write_csv(const string &path, const vector<Row> &rows)
{
    ofstream out(path.c_str());
    for (size_t i = 0; i < kHeaders.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << kHeaders[i];
    }
    out << "\n";

    for (size_t r = 0; r < rows.size(); ++r)
    {
        for (size_t i = 0; i < kHeaders.size(); ++i)
        {
            if (i > 0)
            {
                out << ",";
            }
            Row::const_iterator it = rows[r].find(kHeaders[i]);
            if (it != rows[r].end())
            {
                out << csv_escape(it->second);
            }
        }
        out << "\n";
    }
}

Row base_row(int step_index, int attempt_index, double t_before)
{
    Row row;
    set_value(row, "source", "flowstar");
    set_value(row, "mode", "accepted_step_probe");
    set_value(row, "step_index", format_size(static_cast<size_t>(step_index)));
    set_value(row, "adaptive_attempt_index", format_size(static_cast<size_t>(attempt_index)));
    set_value(row, "t_before", t_before);
    return row;
}

void set_scalars(Row &row, const vector<Real> &scalars)
{
    if (scalars.size() > 0)
    {
        set_value(row, "scalar_x", real_to_double(scalars[0]));
    }
    if (scalars.size() > 1)
    {
        set_value(row, "scalar_y", real_to_double(scalars[1]));
    }
}

void set_target(Row &row, const vector<Interval> &target)
{
    set_widths(row, "target_remainder", target);
}

void set_residual_ratios(Row &row, const vector<Interval> &residual, const vector<Interval> &target)
{
    double total_residual = 0.0;
    double total_target = 0.0;
    if (residual.size() > 0 && target.size() > 0)
    {
        const double rw = interval_width(residual[0]);
        const double tw = interval_width(target[0]);
        set_value(row, "residual_over_target_x", tw == 0.0 ? numeric_limits<double>::quiet_NaN() : rw / tw);
        total_residual += rw;
        total_target += tw;
    }
    if (residual.size() > 1 && target.size() > 1)
    {
        const double rw = interval_width(residual[1]);
        const double tw = interval_width(target[1]);
        set_value(row, "residual_over_target_y", tw == 0.0 ? numeric_limits<double>::quiet_NaN() : rw / tw);
        total_residual += rw;
        total_target += tw;
    }
    set_value(row, "residual_over_target_sum", total_target == 0.0 ? numeric_limits<double>::quiet_NaN() : total_residual / total_target);
}

vector<Interval> target_remainders(const Taylor_Model_Setting &tm_setting, unsigned int dim)
{
    vector<Interval> target;
    for (unsigned int i = 0; i < dim; ++i)
    {
        target.push_back(tm_setting.remainder_estimation[i]);
    }
    return target;
}

int traced_advance_adaptive_symbolic(
    Flowpipe &result,
    const Flowpipe &current,
    const vector<Expression<Real> > &ode,
    const double new_stepsize,
    Taylor_Model_Setting &tm_setting,
    const Global_Setting &g_setting,
    Symbolic_Remainder &symbolic_remainder,
    vector<Row> &rows,
    int step_index,
    double t_before)
{
    const unsigned int rangeDim = ode.size();
    const unsigned int rangeDimExt = rangeDim + 1;
    result.clear();

    TaylorModelVec<Real> tmv_of_x0;
    current.tmvPre.evaluate_time(tmv_of_x0, tm_setting.step_end_exp_table);

    vector<Real> const_of_x0;
    tmv_of_x0.constant(const_of_x0);
    TaylorModelVec<Real> tmv_c0(const_of_x0, rangeDimExt);
    tmv_of_x0.rmConstant();

    TaylorModelVec<Real> x0_linear, x0_other;
    tmv_of_x0.decompose(x0_linear, x0_other);

    Matrix<Real> Phi_L_i(rangeDim, rangeDim);
    x0_linear.linearCoefficients(Phi_L_i);
    Matrix<Real> linear_x0 = Phi_L_i;
    Phi_L_i.right_scale_assign(symbolic_remainder.scalars);

    Matrix<Interval> J_i(rangeDim, 1);

    for (unsigned int i = 1; i < symbolic_remainder.Phi_L.size(); ++i)
    {
        symbolic_remainder.Phi_L[i] = Phi_L_i * symbolic_remainder.Phi_L[i];
    }

    symbolic_remainder.Phi_L.push_back(Phi_L_i);

    for (unsigned int i = 1; i < symbolic_remainder.Phi_L.size(); ++i)
    {
        J_i += symbolic_remainder.Phi_L[i] * symbolic_remainder.J[i - 1];
    }

    Matrix<Interval> J_ip1(rangeDim, 1);
    vector<Interval> range_of_x0;
    vector<Interval> tmv_poly_range;
    current.tmv.polyRangeNormal(tmv_poly_range, tm_setting.step_end_exp_table);

    if (symbolic_remainder.J.size() > 0)
    {
        x0_other.insert_ctrunc_normal(result.tmv, current.tmv, tmv_poly_range, tm_setting.step_end_exp_table, current.domain.size(), tm_setting.order, tm_setting.cutoff_threshold);
        result.tmv.Remainder(J_ip1);

        vector<Polynomial<Real> > poly_tmv;
        current.tmv.Expansion(poly_tmv);
        vector<Polynomial<Real> > linear_part = linear_x0 * poly_tmv;

        for (int i = 0; i < static_cast<int>(rangeDim); ++i)
        {
            result.tmv.tms[i].expansion += linear_part[i];
            result.tmv.tms[i].remainder = J_ip1[i][0] + J_i[i][0];
        }
        result.tmv.intEvalNormal(range_of_x0, tm_setting.step_end_exp_table);
    }
    else
    {
        tmv_of_x0.insert_ctrunc_normal(result.tmv, current.tmv, tmv_poly_range, tm_setting.step_end_exp_table, current.domain.size(), tm_setting.order, tm_setting.cutoff_threshold);
        result.tmv.intEvalNormal(range_of_x0, tm_setting.step_end_exp_table);
        result.tmv.Remainder(J_ip1);
    }

    symbolic_remainder.J.push_back(J_ip1);

    vector<Real> S;
    vector<Real> invS;
    for (int i = 0; i < static_cast<int>(rangeDim); ++i)
    {
        Real sup;
        range_of_x0[i].mag(sup);
        if (sup == 0)
        {
            S.push_back(0);
            invS.push_back(1);
            symbolic_remainder.scalars[i] = 0;
        }
        else
        {
            S.push_back(sup);
            Real tmp = 1 / sup;
            invS.push_back(tmp);
            symbolic_remainder.scalars[i] = tmp;
        }
    }

    vector<Interval> result_tmv_range_before_scale;
    result.tmv.intEvalNormal(result_tmv_range_before_scale, tm_setting.step_end_exp_table);

    result.tmv.scale_assign(invS);
    Interval init_cft(-INITIAL_SIMP, INITIAL_SIMP);
    result.tmv.cutoff_normal(tm_setting.step_end_exp_table, init_cft);

    vector<Interval> result_tmv_normal_range;
    result.tmv.intEvalNormal(result_tmv_normal_range, tm_setting.step_end_exp_table);

    TaylorModelVec<Real> new_x0(S);
    new_x0 += tmv_c0;
    TaylorModelVec<Real> x = new_x0;

    for (unsigned int i = 1; i <= tm_setting.order; ++i)
    {
        x.Picard_no_remainder_assign(new_x0, ode, rangeDimExt, i, tm_setting.cutoff_threshold);
    }

    vector<Interval> picard_no_remainder_range;
    x.intEvalNormal(picard_no_remainder_range, tm_setting.step_exp_table);

    if (new_stepsize > 0)
    {
        tm_setting.setStepsize(new_stepsize, tm_setting.order);
    }

    for (unsigned int i = 0; i < rangeDim; ++i)
    {
        x.tms[i].remainder = tm_setting.remainder_estimation[i];
    }
    const vector<Interval> target = target_remainders(tm_setting, rangeDim);

    TaylorModelVec<Interval> tmvTmp;
    list<Interval> intermediate_ranges;
    vector<Interval> intDifferences(rangeDim);
    vector<Polynomial<Interval> > polyDifferences;
    bool have_poly_differences = false;
    int attempt_index = 0;

    for (;;)
    {
        ++attempt_index;
        const double h_try = tm_setting.step_exp_table[1].sup();
        bool bfound = true;
        intermediate_ranges.clear();
        tmvTmp.clear();
        tmvTmp = TaylorModelVec<Interval>();
        tmvTmp.tms.clear();
        x.Picard_ctrunc_normal(tmvTmp, new_x0, ode, tm_setting.step_exp_table, rangeDimExt, tm_setting.order, tm_setting.cutoff_threshold, intermediate_ranges, g_setting);

        if (!have_poly_differences)
        {
            polyDifferences.clear();
            for (unsigned int i = 0; i < rangeDim; ++i)
            {
                Polynomial<Interval> polyTmp;
                polyTmp = tmvTmp.tms[i].expansion - x.tms[i].expansion;
                polyDifferences.push_back(polyTmp);
            }
            have_poly_differences = true;
        }

        for (unsigned int i = 0; i < rangeDim; ++i)
        {
            polyDifferences[i].intEvalNormal(intDifferences[i], tm_setting.step_exp_table);
            tmvTmp.tms[i].remainder += intDifferences[i];
            if (!tmvTmp.tms[i].remainder.subseteq(x.tms[i].remainder))
            {
                bfound = false;
            }
        }

        vector<Interval> ctrunc_remainder;
        for (unsigned int i = 0; i < rangeDim; ++i)
        {
            ctrunc_remainder.push_back(tmvTmp.tms[i].remainder);
        }

        Row row = base_row(step_index, attempt_index, t_before);
        set_value(row, "h", h_try);
        set_value(row, "t_after", t_before + h_try);
        set_value(row, "accepted", bfound);
        set_value(row, "rejected", !bfound);
        set_value(row, "status", bfound ? "accepted" : "rejected");
        set_widths(row, "tmv_pre_range", picard_no_remainder_range);
        set_widths(row, "tmv_right_range", result_tmv_range_before_scale);
        set_widths(row, "tmv_right_normal_range", result_tmv_normal_range);
        set_widths(row, "endpoint_pre_center", range_of_x0);
        if (const_of_x0.size() > 0)
        {
            set_value(row, "center_x", real_to_double(const_of_x0[0]));
        }
        if (const_of_x0.size() > 1)
        {
            set_value(row, "center_y", real_to_double(const_of_x0[1]));
        }
        if (S.size() > 0)
        {
            set_value(row, "scale_x", real_to_double(S[0]));
            set_value(row, "inv_scale_x", real_to_double(invS[0]));
        }
        if (S.size() > 1)
        {
            set_value(row, "scale_y", real_to_double(S[1]));
            set_value(row, "inv_scale_y", real_to_double(invS[1]));
        }
        vector<Interval> new_x0_range;
        new_x0.intEvalNormal(new_x0_range, tm_setting.step_end_exp_table);
        set_widths(row, "new_x0", new_x0_range);
        set_target(row, target);
        set_widths(row, "picard_no_remainder_residual", picard_no_remainder_range);
        set_widths(row, "picard_ctrunc_normal_residual", ctrunc_remainder);
        set_widths(row, "cutoff_polynomial_difference", intDifferences);
        set_value(row, "symbolic_J_size", format_size(symbolic_remainder.J.size()));
        set_value(row, "symbolic_Phi_L_size", format_size(symbolic_remainder.Phi_L.size()));
        set_scalars(row, symbolic_remainder.scalars);
        set_widths(row, "symbolic_J", matrix_column_intervals(J_ip1));
        set_widths(row, "symbolic_propagated", matrix_column_intervals(J_i));
        set_widths(row, "residual", ctrunc_remainder);
        set_residual_ratios(row, ctrunc_remainder, target);

        if (!bfound)
        {
            set_value(row, "message", "Picard_ctrunc_normal remainder not contained in target; shrinking h");
            rows.push_back(row);
            const double newStep = h_try * LAMBDA_DOWN;
            if (newStep < tm_setting.step_min)
            {
                return 0;
            }
            tm_setting.setStepsize(newStep, tm_setting.order);
            continue;
        }

        for (unsigned int i = 0; i < rangeDim; ++i)
        {
            x.tms[i].remainder = tmvTmp.tms[i].remainder;
        }

        bool bfinished = false;
        for (int rSteps = 0; !bfinished && (rSteps <= MAX_REFINEMENT_STEPS); ++rSteps)
        {
            bfinished = true;

            vector<Interval> newRemainders;
            x.Picard_ctrunc_normal_remainder(newRemainders, ode, tm_setting.step_exp_table[1], tm_setting.order, intermediate_ranges, g_setting);

            for (unsigned int i = 0; i < rangeDim; ++i)
            {
                newRemainders[i] += intDifferences[i];

                if (newRemainders[i].subseteq(x.tms[i].remainder))
                {
                    if (x.tms[i].remainder.widthRatio(newRemainders[i]) <= STOP_RATIO)
                    {
                        bfinished = false;
                    }

                    x.tms[i].remainder = newRemainders[i];
                }
                else
                {
                    bfinished = true;
                    break;
                }
            }
        }

        result.tmvPre = x;
        result.domain = current.domain;
        result.domain[0] = tm_setting.step_exp_table[1];

        vector<Interval> final_box;
        result.intEvalNormal(final_box, tm_setting.step_exp_table, tm_setting.order, tm_setting.cutoff_threshold);
        set_widths(row, "final_flowpipe", final_box);
        vector<Interval> final_remainders;
        for (unsigned int i = 0; i < rangeDim; ++i)
        {
            final_remainders.push_back(x.tms[i].remainder);
        }
        set_widths(row, "picard_ctrunc_normal_residual", final_remainders);
        set_widths(row, "residual", final_remainders);
        set_residual_ratios(row, final_remainders, target);
        rows.push_back(row);
        return 1;
    }
}

} // namespace

int main(int argc, char **argv)
{
    string output_csv = "flowstar_trace.csv";
    if (argc > 1)
    {
        output_csv = argv[1];
    }
    double horizon = 1.0;
    if (argc > 2)
    {
        horizon = atof(argv[2]);
    }
    int max_segments = 0;
    if (argc > 3)
    {
        max_segments = atoi(argv[3]);
    }

    Variables vars;
    int x_id = vars.declareVar("x");
    int y_id = vars.declareVar("y");
    vars.declareVar("t");

    ODE<Real> ode({"y", "(1 - x^2) * y - x", "1"}, vars);
    Computational_Setting setting(vars);
    setting.printOff();

    vector<Interval> box(vars.size());
    box[x_id] = Interval(1.1, 1.4);
    box[y_id] = Interval(2.35, 2.45);
    Flowpipe initialSet(box);
    Symbolic_Remainder sr(initialSet, 100);

    vector<Row> rows;
    Flowpipe newFlowpipe;
    Flowpipe currentFlowpipe = initialSet;
    vector<Constraint> dummy_invariant;

    setting.tm_setting.setStepsize(setting.tm_setting.step_max, setting.tm_setting.order);
    double new_stepsize = -1;
    double t = THRESHOLD_HIGH;
    int step_index = 0;

    while (t < horizon)
    {
        if (max_segments > 0 && step_index >= max_segments)
        {
            break;
        }
        const int res = traced_advance_adaptive_symbolic(
            newFlowpipe,
            currentFlowpipe,
            ode.expressions,
            new_stepsize,
            setting.tm_setting,
            setting.g_setting,
            sr,
            rows,
            step_index,
            t);
        if (res != 1)
        {
            Row row = base_row(step_index, 0, t);
            set_value(row, "accepted", false);
            set_value(row, "rejected", true);
            set_value(row, "status", "failed");
            set_value(row, "message", "Flow* traced advance failed below minimum step");
            rows.push_back(row);
            break;
        }

        double current_stepsize = setting.tm_setting.step_exp_table[1].sup();
        const double remaining_time = horizon - t;
        if (remaining_time < current_stepsize)
        {
            current_stepsize = remaining_time;
            newFlowpipe.domain[0].setSup(remaining_time);
            if (!rows.empty())
            {
                set_value(rows.back(), "h", current_stepsize);
                set_value(rows.back(), "t_after", t + current_stepsize);
            }
        }

        currentFlowpipe = newFlowpipe;
        if (sr.J.size() >= sr.max_size)
        {
            sr.reset(currentFlowpipe.tmvPre.tms.size());
        }

        t += current_stepsize;
        new_stepsize = current_stepsize * LAMBDA_UP;
        if (new_stepsize > setting.tm_setting.step_max - THRESHOLD_HIGH)
        {
            new_stepsize = -1;
        }
        ++step_index;
    }

    write_csv(output_csv, rows);
    return 0;
}
