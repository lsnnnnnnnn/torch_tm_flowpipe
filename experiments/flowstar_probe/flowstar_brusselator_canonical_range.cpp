#include "Continuous.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

const char *const kSchema = "torch_tm_flowpipe.brusselator_live_range_exchange/1";
const char *const kOutputSchema = "flowstar.brusselator_canonical_composition/1";

string unsigned_text(const unsigned long value)
{
    ostringstream output;
    output << value;
    return output.str();
}

map<string, string> read_fields(const string &path)
{
    ifstream input(path.c_str(), ios::binary);
    if (!input)
    {
        throw runtime_error("cannot open canonical object: " + path);
    }
    map<string, string> fields;
    string line;
    unsigned long line_number = 0;
    while (getline(input, line))
    {
        ++line_number;
        if (line.empty() || line.find('\r') != string::npos)
        {
            throw runtime_error("empty/CR canonical record at line " + unsigned_text(line_number));
        }
        const size_t separator = line.find('=');
        if (separator == string::npos || separator == 0 || line.find('=', separator + 1) != string::npos)
        {
            throw runtime_error("invalid canonical key=value record at line " + unsigned_text(line_number));
        }
        const string key = line.substr(0, separator);
        const string value = line.substr(separator + 1);
        if (value.empty() || !fields.insert(make_pair(key, value)).second)
        {
            throw runtime_error("empty or duplicate canonical field: " + key);
        }
    }
    return fields;
}

string require(const map<string, string> &fields, const string &key)
{
    map<string, string>::const_iterator iterator = fields.find(key);
    if (iterator == fields.end())
    {
        throw runtime_error("missing canonical field: " + key);
    }
    return iterator->second;
}

unsigned long parse_unsigned(const string &text, const string &key)
{
    if (text.empty() || text[0] == '-')
    {
        throw runtime_error("invalid unsigned field: " + key);
    }
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
    {
        throw runtime_error("invalid unsigned field: " + key);
    }
    return value;
}

double parse_hex(const string &text, const string &key)
{
    if (text.find("0x") == string::npos)
    {
        throw runtime_error("non-hexadecimal numeric field: " + key);
    }
    char *end = NULL;
    errno = 0;
    const double value = strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !isfinite(value))
    {
        throw runtime_error("invalid/nonfinite binary64 field: " + key);
    }
    return value;
}

Interval take_interval(const map<string, string> &fields, const string &key)
{
    const double lower = parse_hex(require(fields, key + ".lo"), key + ".lo");
    const double upper = parse_hex(require(fields, key + ".hi"), key + ".hi");
    if (lower > upper)
    {
        throw runtime_error("inverted canonical interval: " + key);
    }
    return Interval(lower, upper);
}

vector<unsigned int> parse_exponents(
    const string &text,
    const unsigned long expected,
    const string &key)
{
    vector<unsigned int> result;
    string current;
    for (size_t index = 0; index <= text.size(); ++index)
    {
        if (index == text.size() || text[index] == ',')
        {
            result.push_back(static_cast<unsigned int>(parse_unsigned(current, key)));
            current.clear();
        }
        else
        {
            current.push_back(text[index]);
        }
    }
    if (result.size() != expected)
    {
        throw runtime_error("wrong exponent dimension: " + key);
    }
    return result;
}

vector<unsigned int> flowstar_exponents(const vector<unsigned int> &canonical)
{
    if (canonical.size() == 2)
    {
        vector<unsigned int> result(3, 0);
        result[1] = canonical[0];
        result[2] = canonical[1];
        return result;
    }
    if (canonical.size() == 3)
    {
        vector<unsigned int> result(3, 0);
        result[0] = canonical[2];
        result[1] = canonical[0];
        result[2] = canonical[1];
        return result;
    }
    throw runtime_error("canonical TM must have ux,uy or ux,uy,tau variables");
}

TaylorModelVec<Real> take_tmv(const map<string, string> &fields, const string &prefix)
{
    const unsigned long component_count = parse_unsigned(
        require(fields, prefix + ".component_count"), prefix + ".component_count");
    const unsigned long variable_count = parse_unsigned(
        require(fields, prefix + ".variable_count"), prefix + ".variable_count");
    const string variable_order = require(fields, prefix + ".variable_order");
    if ((variable_count == 2 && variable_order != "ux,uy") ||
        (variable_count == 3 && variable_order != "ux,uy,tau"))
    {
        throw runtime_error("unsupported canonical variable order: " + prefix);
    }
    const unsigned long domain_count = parse_unsigned(
        require(fields, prefix + ".domain_count"), prefix + ".domain_count");
    if (domain_count != variable_count)
    {
        throw runtime_error("canonical TM domain dimension mismatch: " + prefix);
    }
    for (unsigned long variable = 0; variable < domain_count; ++variable)
    {
        (void)take_interval(fields, prefix + ".domain." + unsigned_text(variable));
    }
    vector<TaylorModel<Real> > models;
    for (unsigned long component = 0; component < component_count; ++component)
    {
        const string base = prefix + ".component." + unsigned_text(component);
        const unsigned long order = parse_unsigned(require(fields, base + ".order"), base + ".order");
        const unsigned long term_count = parse_unsigned(
            require(fields, base + ".term_count"), base + ".term_count");
        list<Term<Real> > terms;
        for (unsigned long term_index = 0; term_index < term_count; ++term_index)
        {
            const string term = base + ".term." + unsigned_text(term_index);
            const vector<unsigned int> canonical = parse_exponents(
                require(fields, term + ".exponents"), variable_count, term + ".exponents");
            unsigned long degree = 0;
            for (size_t index = 0; index < canonical.size(); ++index)
            {
                degree += canonical[index];
            }
            if (degree != parse_unsigned(require(fields, term + ".total_degree"), term + ".total_degree") ||
                degree > order)
            {
                throw runtime_error("canonical term degree/order mismatch: " + term);
            }
            const double coefficient = parse_hex(
                require(fields, term + ".coefficient_hex"), term + ".coefficient_hex");
            if (coefficient == 0.0)
            {
                throw runtime_error("explicit zero canonical coefficient: " + term);
            }
            Term<Real> parsed(Real(coefficient), 3);
            parsed.canonicalRangeSetDegrees(flowstar_exponents(canonical));
            terms.push_back(parsed);
        }
        terms.sort();
        const Interval remainder = take_interval(fields, base + ".ordinary_remainder");
        models.push_back(TaylorModel<Real>(Polynomial<Real>(terms), remainder));
    }
    return TaylorModelVec<Real>(models);
}

vector<Interval> take_interval_vector(const map<string, string> &fields, const string &prefix)
{
    const unsigned long count = parse_unsigned(require(fields, prefix + ".count"), prefix + ".count");
    vector<Interval> result;
    for (unsigned long index = 0; index < count; ++index)
    {
        result.push_back(take_interval(fields, prefix + "." + unsigned_text(index)));
    }
    return result;
}

vector<Real> take_real_vector(const map<string, string> &fields, const string &prefix)
{
    const unsigned long count = parse_unsigned(require(fields, prefix + ".count"), prefix + ".count");
    vector<Real> result;
    for (unsigned long index = 0; index < count; ++index)
    {
        const string key = prefix + "." + unsigned_text(index);
        result.push_back(Real(parse_hex(require(fields, key), key)));
    }
    return result;
}

Matrix<Real> take_real_matrix(const map<string, string> &fields, const string &prefix)
{
    const unsigned long rows = parse_unsigned(require(fields, prefix + ".rows"), prefix + ".rows");
    const unsigned long cols = parse_unsigned(require(fields, prefix + ".cols"), prefix + ".cols");
    Matrix<Real> result(static_cast<int>(rows), static_cast<int>(cols));
    for (unsigned long row = 0; row < rows; ++row)
    {
        for (unsigned long column = 0; column < cols; ++column)
        {
            const string key = prefix + "." + unsigned_text(row) + "." + unsigned_text(column);
            result[row][column] = Real(parse_hex(require(fields, key), key));
        }
    }
    return result;
}

string hex_double(const double value)
{
    if (value == 0.0)
    {
        return signbit(value) ? "-0x0.0p+0" : "0x0.0p+0";
    }
    char buffer[64];
    if (snprintf(buffer, sizeof(buffer), "%.13a", value) <= 0)
    {
        throw runtime_error("cannot encode binary64 hexadecimal value");
    }
    return string(buffer);
}

void write_interval_json(ostream &output, const Interval &value)
{
    output << "{\"lo\":" << setprecision(17) << value.inf();
    output << ",\"hi\":" << setprecision(17) << value.sup();
    output << ",\"lo_hex\":\"" << hex_double(value.inf()) << "\"";
    output << ",\"hi_hex\":\"" << hex_double(value.sup()) << "\"}";
}

void write_box_json(ostream &output, const char *key, const vector<Interval> &box, bool &first)
{
    if (!first) output << ',';
    first = false;
    output << '\"' << key << "\":[";
    for (size_t index = 0; index < box.size(); ++index)
    {
        if (index > 0) output << ',';
        write_interval_json(output, box[index]);
    }
    output << ']';
}

void write_term_counts_json(
    ostream &output,
    const char *key,
    const TaylorModelVec<Real> &value,
    bool &first)
{
    if (!first) output << ',';
    first = false;
    output << '\"' << key << "\":[";
    for (size_t component = 0; component < value.tms.size(); ++component)
    {
        if (component > 0) output << ',';
        output << value.tms[component].expansion.terms.size();
    }
    output << ']';
}

string degree_key(const vector<unsigned int> &degrees)
{
    ostringstream output;
    for (size_t index = 0; index < degrees.size(); ++index)
    {
        if (index > 0) output << ',';
        output << degrees[index];
    }
    return output.str();
}

void write_removed_terms_json(
    ostream &output,
    const char *key,
    const TaylorModelVec<Real> &before,
    const TaylorModelVec<Real> &after,
    const vector<Interval> &endpoint_table,
    bool &first)
{
    if (!first) output << ',';
    first = false;
    output << '\"' << key << "\":[";
    bool first_term = true;
    for (size_t component = 0; component < before.tms.size(); ++component)
    {
        set<string> retained;
        for (list<Term<Real> >::const_iterator iterator = after.tms[component].expansion.terms.begin();
             iterator != after.tms[component].expansion.terms.end(); ++iterator)
        {
            retained.insert(degree_key(iterator->canonicalRangeDegrees()));
        }
        for (list<Term<Real> >::const_iterator iterator = before.tms[component].expansion.terms.begin();
             iterator != before.tms[component].expansion.terms.end(); ++iterator)
        {
            const vector<unsigned int> &degrees = iterator->canonicalRangeDegrees();
            if (retained.find(degree_key(degrees)) != retained.end()) continue;
            Interval payment;
            iterator->intEvalNormal(payment, endpoint_table);
            if (!first_term) output << ',';
            first_term = false;
            output << "{\"component\":" << component;
            output << ",\"exponents_tau_ux_uy\":[" << degrees[0] << ',' << degrees[1] << ',' << degrees[2] << ']';
            output << ",\"coefficient_hex\":\"" << hex_double(iterator->canonicalRangeCoefficient().toDouble()) << "\"";
            output << ",\"interval_payment\":";
            write_interval_json(output, payment);
            output << '}';
        }
    }
    output << ']';
}

void write_real_hex(ostream &output, const string &key, const Real &value)
{
    output << key << '=' << hex_double(value.toDouble()) << '\n';
}

void write_interval_records(ostream &output, const string &key, const Interval &value)
{
    output << key << ".lo=" << hex_double(value.inf()) << '\n';
    output << key << ".hi=" << hex_double(value.sup()) << '\n';
}

void write_composed_tmv(ostream &output, const TaylorModelVec<Real> &value)
{
    output << "tm.flowstar_inserted.component_count=" << value.tms.size() << '\n';
    output << "tm.flowstar_inserted.variable_count=2\n";
    output << "tm.flowstar_inserted.variable_order=ux,uy\n";
    output << "tm.flowstar_inserted.domain_count=2\n";
    output << "tm.flowstar_inserted.domain.0.lo=-0x1.0000000000000p+0\n";
    output << "tm.flowstar_inserted.domain.0.hi=0x1.0000000000000p+0\n";
    output << "tm.flowstar_inserted.domain.1.lo=-0x1.0000000000000p+0\n";
    output << "tm.flowstar_inserted.domain.1.hi=0x1.0000000000000p+0\n";
    for (size_t component = 0; component < value.tms.size(); ++component)
    {
        const TaylorModel<Real> &model = value.tms[component];
        const string base = "tm.flowstar_inserted.component." + unsigned_text(component);
        output << base << ".order=6\n";
        size_t retained = 0;
        for (list<Term<Real> >::const_iterator iterator = model.expansion.terms.begin();
             iterator != model.expansion.terms.end(); ++iterator)
        {
            const vector<unsigned int> &degrees = iterator->canonicalRangeDegrees();
            if (degrees.size() != 3 || degrees[0] != 0)
            {
                throw runtime_error("Flow* boundary composition unexpectedly depends on local time");
            }
            ++retained;
        }
        output << base << ".term_count=" << retained << '\n';
        size_t term_index = 0;
        for (list<Term<Real> >::const_iterator iterator = model.expansion.terms.begin();
             iterator != model.expansion.terms.end(); ++iterator, ++term_index)
        {
            const vector<unsigned int> &degrees = iterator->canonicalRangeDegrees();
            const string term = base + ".term." + unsigned_text(term_index);
            output << term << ".exponents=" << degrees[1] << ',' << degrees[2] << '\n';
            output << term << ".total_degree=" << iterator->degree() << '\n';
            write_real_hex(output, term + ".coefficient_hex", iterator->canonicalRangeCoefficient());
        }
        write_interval_records(output, base + ".ordinary_remainder", model.remainder);
    }
}

vector<Interval> remainders(const TaylorModelVec<Real> &value)
{
    Matrix<Interval> matrix(static_cast<int>(value.tms.size()), 1);
    value.Remainder(matrix);
    vector<Interval> result;
    for (size_t component = 0; component < value.tms.size(); ++component)
    {
        result.push_back(matrix[component][0]);
    }
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
        {
            throw runtime_error("usage: flowstar_brusselator_canonical_range INPUT OUTPUT");
        }
        const map<string, string> fields = read_fields(argv[1]);
        if (require(fields, "schema") != kSchema)
        {
            throw runtime_error("canonical exchange schema mismatch");
        }
        if (require(fields, "polynomial_variable_order") != "ux,uy,tau" ||
            require(fields, "flowstar_harness_variable_order") != "tau,ux,uy" ||
            parse_unsigned(require(fields, "state_dimension"), "state_dimension") != 2 ||
            parse_unsigned(require(fields, "order"), "order") != 6 ||
            parse_unsigned(require(fields, "tau_index"), "tau_index") != 2)
        {
            throw runtime_error("frozen canonical dimension/order contract mismatch");
        }
        const unsigned long accepted_step = parse_unsigned(require(fields, "accepted_step"), "accepted_step");
        const double cutoff_value = parse_hex(require(fields, "cutoff_threshold_hex"), "cutoff_threshold_hex");
        const Interval cutoff(-cutoff_value, cutoff_value);
        const vector<Interval> tube_table = take_interval_vector(fields, "table.step_exp");
        const vector<Real> endpoint_real = take_real_vector(fields, "table.step_end_exp");
        vector<Interval> endpoint_table;
        for (size_t index = 0; index < endpoint_real.size(); ++index)
        {
            endpoint_table.push_back(Interval(endpoint_real[index]));
        }

        TaylorModelVec<Real> tube = take_tmv(fields, "tm.segment_tube");
        TaylorModelVec<Real> endpoint_pre_cutoff = take_tmv(fields, "tm.segment_endpoint_pre_cutoff");
        TaylorModelVec<Real> endpoint = take_tmv(fields, "tm.segment_endpoint_raw");
        TaylorModelVec<Real> endpoint_flowstar_cutoff = endpoint_pre_cutoff;
        endpoint_flowstar_cutoff.cutoff_normal(endpoint_table, cutoff);
        TaylorModelVec<Real> endpoint_poly = endpoint;
        TaylorModelVec<Real> tube_poly = tube;
        for (size_t component = 0; component < endpoint_poly.tms.size(); ++component)
        {
            endpoint_poly.tms[component].remainder = Interval(0.0);
        }
        for (size_t component = 0; component < tube_poly.tms.size(); ++component)
        {
            tube_poly.tms[component].remainder = Interval(0.0);
        }
        vector<Interval> endpoint_poly_range;
        vector<Interval> endpoint_full_range;
        vector<Interval> tube_poly_range;
        vector<Interval> tube_full_range;
        vector<Interval> endpoint_pre_cutoff_range;
        vector<Interval> endpoint_flowstar_cutoff_range;
        endpoint_poly.intEvalNormal(endpoint_poly_range, endpoint_table);
        endpoint.intEvalNormal(endpoint_full_range, endpoint_table);
        tube_poly.intEvalNormal(tube_poly_range, tube_table);
        tube.intEvalNormal(tube_full_range, tube_table);
        endpoint_pre_cutoff.intEvalNormal(endpoint_pre_cutoff_range, endpoint_table);
        endpoint_flowstar_cutoff.intEvalNormal(endpoint_flowstar_cutoff_range, endpoint_table);

        TaylorModelVec<Real> right = take_tmv(fields, "tm.right_map_input");
        vector<Interval> right_poly_range;
        right.polyRangeNormal(right_poly_range, endpoint_table);
        TaylorModelVec<Real> composed;
        vector<Interval> current_owner;
        const string branch = require(fields, "boundary.composition_branch");
        if (branch == "full_reanchor")
        {
            TaylorModelVec<Real> outer = take_tmv(fields, "tm.boundary_outer_full");
            outer.insert_ctrunc_normal(composed, right, right_poly_range, endpoint_table, 3, 6, cutoff);
            current_owner = remainders(composed);
        }
        else if (branch == "nonlinear_plus_linear_queue")
        {
            TaylorModelVec<Real> nonlinear = take_tmv(fields, "tm.boundary_outer_nonlinear");
            nonlinear.insert_ctrunc_normal(composed, right, right_poly_range, endpoint_table, 3, 6, cutoff);
            current_owner = remainders(composed);
            Matrix<Real> linear = take_real_matrix(fields, "boundary.linear");
            vector<Polynomial<Real> > right_polynomials;
            right.Expansion(right_polynomials);
            const vector<Polynomial<Real> > linear_part = linear * right_polynomials;
            if (linear_part.size() != composed.tms.size())
            {
                throw runtime_error("linear recomposition dimension mismatch");
            }
            for (size_t component = 0; component < composed.tms.size(); ++component)
            {
                composed.tms[component].expansion += linear_part[component];
            }
            const vector<Interval> propagated = take_interval_vector(fields, "boundary.sr_propagated_history");
            if (propagated.size() != composed.tms.size())
            {
                throw runtime_error("propagated history dimension mismatch");
            }
            for (size_t component = 0; component < composed.tms.size(); ++component)
            {
                composed.tms[component].remainder += propagated[component];
            }
        }
        else
        {
            throw runtime_error("unknown accepted-boundary composition branch");
        }
        vector<Interval> composed_poly_range;
        vector<Interval> composed_full_range;
        TaylorModelVec<Real> composed_poly = composed;
        for (size_t component = 0; component < composed_poly.tms.size(); ++component)
        {
            composed_poly.tms[component].remainder = Interval(0.0);
        }
        composed_poly.intEvalNormal(composed_poly_range, endpoint_table);
        composed.intEvalNormal(composed_full_range, endpoint_table);

        TaylorModelVec<Real> torch_inserted = take_tmv(fields, "tm.boundary_torch_inserted");
        TaylorModelVec<Real> torch_post_right = take_tmv(fields, "tm.right_map_torch_post_cutoff");
        const vector<Real> post_scales = take_real_vector(fields, "post.scale");
        vector<Real> inverse_scales;
        for (size_t component = 0; component < post_scales.size(); ++component)
        {
            if (post_scales[component] == 0)
            {
                inverse_scales.push_back(Real(1.0));
            }
            else
            {
                Real inverse = 1 / post_scales[component];
                inverse_scales.push_back(inverse);
            }
        }
        TaylorModelVec<Real> flow_scaled_pre_cutoff = torch_inserted;
        flow_scaled_pre_cutoff.scale_assign(inverse_scales);
        TaylorModelVec<Real> flow_initial_simp = flow_scaled_pre_cutoff;
        flow_initial_simp.cutoff_normal(endpoint_table, Interval(-1e-4, 1e-4));
        vector<Interval> flow_scaled_pre_cutoff_range;
        vector<Interval> flow_initial_simp_range;
        vector<Interval> torch_post_right_range;
        flow_scaled_pre_cutoff.intEvalNormal(flow_scaled_pre_cutoff_range, endpoint_table);
        flow_initial_simp.intEvalNormal(flow_initial_simp_range, endpoint_table);
        torch_post_right.intEvalNormal(torch_post_right_range, endpoint_table);
        const vector<Interval> flow_scaled_pre_cutoff_remainders = remainders(flow_scaled_pre_cutoff);
        const vector<Interval> flow_initial_simp_remainders = remainders(flow_initial_simp);
        const vector<Interval> torch_post_right_remainders = remainders(torch_post_right);

        ofstream output(argv[2], ios::binary);
        if (!output)
        {
            throw runtime_error("cannot write Flow* canonical composition output");
        }
        output << "schema=" << kOutputSchema << '\n';
        output << "accepted_step=" << accepted_step << '\n';
        output << "source.flowstar_commit=" << require(fields, "source.flowstar_commit") << '\n';
        output << "source.input_checkpoint_sha256=" << require(fields, "checkpoint_sha256") << '\n';
        output << "boundary.composition_branch=" << branch << '\n';
        write_composed_tmv(output, composed);
        output << "boundary.current_owner.count=" << current_owner.size() << '\n';
        for (size_t component = 0; component < current_owner.size(); ++component)
        {
            write_interval_records(
                output,
                "boundary.current_owner." + unsigned_text(component),
                current_owner[component]);
        }
        output.close();

        bool first = true;
        cout << '{';
        cout << "\"schema\":\"flowstar.brusselator_canonical_range_result/1\"";
        first = false;
        cout << ",\"accepted_step\":" << accepted_step;
        write_box_json(cout, "endpoint_polynomial", endpoint_poly_range, first);
        write_box_json(cout, "endpoint_full", endpoint_full_range, first);
        write_box_json(cout, "tube_polynomial", tube_poly_range, first);
        write_box_json(cout, "tube_full", tube_full_range, first);
        write_box_json(cout, "endpoint_pre_cutoff_full", endpoint_pre_cutoff_range, first);
        write_box_json(cout, "endpoint_flowstar_cutoff_full", endpoint_flowstar_cutoff_range, first);
        write_box_json(cout, "composition_polynomial", composed_poly_range, first);
        write_box_json(cout, "composition_full", composed_full_range, first);
        write_box_json(cout, "composition_current_owner", current_owner, first);
        write_box_json(cout, "right_map_flow_scaled_pre_cutoff_full", flow_scaled_pre_cutoff_range, first);
        write_box_json(cout, "right_map_flow_initial_simp_full", flow_initial_simp_range, first);
        write_box_json(cout, "right_map_torch_actual_full", torch_post_right_range, first);
        write_box_json(cout, "right_map_flow_scaled_pre_cutoff_remainder", flow_scaled_pre_cutoff_remainders, first);
        write_box_json(cout, "right_map_flow_initial_simp_remainder", flow_initial_simp_remainders, first);
        write_box_json(cout, "right_map_torch_actual_remainder", torch_post_right_remainders, first);
        write_term_counts_json(cout, "right_map_flow_scaled_pre_cutoff_term_counts", flow_scaled_pre_cutoff, first);
        write_term_counts_json(cout, "right_map_flow_initial_simp_term_counts", flow_initial_simp, first);
        write_term_counts_json(cout, "right_map_torch_actual_term_counts", torch_post_right, first);
        write_removed_terms_json(
            cout,
            "right_map_flow_initial_simp_removed_terms",
            flow_scaled_pre_cutoff,
            flow_initial_simp,
            endpoint_table,
            first);
        cout << "}\n";
        return 0;
    }
    catch (const exception &error)
    {
        cerr << error.what() << '\n';
        return 2;
    }
}
