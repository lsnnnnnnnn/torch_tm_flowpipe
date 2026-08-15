#include "Continuous.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace flowstar;
using namespace std;

namespace
{

const char *const kSchema = "flowstar_lossless_state_queue_v1";
const double kStep = 0.01;
const unsigned int kOrder = 4;

struct Snapshot
{
    string producer;
    string phase;
    unsigned long step;
    Real local_time;
    Flowpipe flowpipe;
    Symbolic_Remainder queue;
};

string canonical_real(const Real &value)
{
    const string encoded = value.auditCanonicalBinary();
    if (encoded.empty())
    {
        throw runtime_error("canonical schema rejects NaN/Inf");
    }
    return encoded;
}

Real parse_real(const string &encoded)
{
    Real value;
    if (!value.auditSetCanonicalBinary(encoded))
    {
        throw runtime_error("invalid canonical MPFR value: " + encoded);
    }
    return value;
}

void write_real(ostream &output, const string &key, const Real &value)
{
    output << key << '=' << canonical_real(value) << '\n';
}

void write_interval(ostream &output, const string &key, const Interval &value)
{
    Real lower;
    Real upper;
    value.inf(lower);
    value.sup(upper);
    write_real(output, key + ".lo", lower);
    write_real(output, key + ".hi", upper);
}

string unsigned_text(const unsigned long value)
{
    ostringstream output;
    output << value;
    return output.str();
}

unsigned long parse_unsigned(const string &text, const string &key)
{
    if (text.empty() || text[0] == '-')
    {
        throw runtime_error("invalid unsigned field " + key);
    }
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text.c_str(), &end, 10);
    if (errno != 0 || *end != '\0')
    {
        throw runtime_error("invalid unsigned field " + key);
    }
    return value;
}

map<string, string> read_fields(const string &path)
{
    ifstream input(path.c_str());
    if (!input)
    {
        throw runtime_error("cannot open schema input: " + path);
    }
    map<string, string> fields;
    string line;
    unsigned long line_number = 0;
    while (getline(input, line))
    {
        ++line_number;
        if (line.empty() || line.find('\r') != string::npos)
        {
            throw runtime_error("empty/CR line in canonical schema");
        }
        const size_t separator = line.find('=');
        if (separator == string::npos || separator == 0 || line.find('=', separator + 1) != string::npos)
        {
            throw runtime_error("invalid key=value record at line " + unsigned_text(line_number));
        }
        const string key = line.substr(0, separator);
        const string value = line.substr(separator + 1);
        if (value.empty() || !fields.insert(make_pair(key, value)).second)
        {
            throw runtime_error("empty or duplicate schema field: " + key);
        }
    }
    return fields;
}

string take(map<string, string> &fields, const string &key)
{
    map<string, string>::iterator iterator = fields.find(key);
    if (iterator == fields.end())
    {
        throw runtime_error("missing schema field: " + key);
    }
    const string value = iterator->second;
    fields.erase(iterator);
    return value;
}

Interval take_interval(map<string, string> &fields, const string &key)
{
    const Real lower = parse_real(take(fields, key + ".lo"));
    const Real upper = parse_real(take(fields, key + ".hi"));
    if (lower > upper)
    {
        throw runtime_error("inverted interval: " + key);
    }
    return Interval(lower, upper, 0);
}

string join_degrees(const vector<unsigned int> &degrees)
{
    ostringstream output;
    for (size_t index = 0; index < degrees.size(); ++index)
    {
        if (index > 0)
        {
            output << ',';
        }
        output << degrees[index];
    }
    return output.str();
}

vector<unsigned int> parse_degrees(
    const string &text,
    const unsigned long expected_dimension,
    const string &key)
{
    vector<unsigned int> result;
    string current;
    for (size_t index = 0; index <= text.size(); ++index)
    {
        if (index == text.size() || text[index] == ',')
        {
            const unsigned long value = parse_unsigned(current, key);
            result.push_back(static_cast<unsigned int>(value));
            current.clear();
        }
        else
        {
            current.push_back(text[index]);
        }
    }
    if (result.size() != expected_dimension)
    {
        throw runtime_error("wrong exponent-vector dimension: " + key);
    }
    return result;
}

void write_tmv(ostream &output, const string &prefix, const TaylorModelVec<Real> &tmv)
{
    output << prefix << ".component_count=" << tmv.tms.size() << '\n';
    for (size_t component = 0; component < tmv.tms.size(); ++component)
    {
        const TaylorModel<Real> &model = tmv.tms[component];
        const string base = prefix + ".component." + unsigned_text(component);
        output << base << ".term_count=" << model.expansion.terms.size() << '\n';
        size_t term_index = 0;
        for (list<Term<Real> >::const_iterator iterator = model.expansion.terms.begin();
             iterator != model.expansion.terms.end(); ++iterator, ++term_index)
        {
            const string term = base + ".term." + unsigned_text(term_index);
            output << term << ".exponents=" << join_degrees(iterator->auditDegrees()) << '\n';
            output << term << ".total_degree=" << iterator->degree() << '\n';
            write_real(output, term + ".coefficient", iterator->auditCoefficient());
        }
        write_interval(output, base + ".remainder", model.remainder);
    }
}

TaylorModelVec<Real> take_tmv(
    map<string, string> &fields,
    const string &prefix,
    const unsigned long state_dimension,
    const unsigned long variable_dimension,
    const unsigned long order)
{
    const unsigned long count = parse_unsigned(take(fields, prefix + ".component_count"), prefix);
    if (count != state_dimension)
    {
        throw runtime_error("wrong component count: " + prefix);
    }
    vector<TaylorModel<Real> > models;
    for (unsigned long component = 0; component < count; ++component)
    {
        const string base = prefix + ".component." + unsigned_text(component);
        const unsigned long term_count = parse_unsigned(take(fields, base + ".term_count"), base);
        list<Term<Real> > terms;
        vector<unsigned int> previous;
        for (unsigned long term_index = 0; term_index < term_count; ++term_index)
        {
            const string term = base + ".term." + unsigned_text(term_index);
            const vector<unsigned int> exponents = parse_degrees(
                take(fields, term + ".exponents"), variable_dimension, term);
            const unsigned long declared_degree = parse_unsigned(
                take(fields, term + ".total_degree"), term);
            unsigned long degree = 0;
            for (size_t index = 0; index < exponents.size(); ++index)
            {
                degree += exponents[index];
            }
            if (degree != declared_degree || degree > order)
            {
                throw runtime_error("wrong total degree/order: " + term);
            }
            if (!previous.empty() && exponents == previous)
            {
                throw runtime_error("duplicate adjacent exponent vector: " + term);
            }
            previous = exponents;
            const Real coefficient = parse_real(take(fields, term + ".coefficient"));
            if (coefficient == 0.0)
            {
                throw runtime_error("explicit zero coefficient is non-canonical: " + term);
            }
            terms.push_back(Term<Real>::auditCreate(coefficient, exponents));
        }
        const Interval remainder = take_interval(fields, base + ".remainder");
        const Polynomial<Real> polynomial(terms);
        models.push_back(TaylorModel<Real>(polynomial, remainder));
    }
    return TaylorModelVec<Real>(models);
}

void write_matrix_interval(ostream &output, const string &prefix, const Matrix<Interval> &matrix)
{
    output << prefix << ".rows=" << matrix.rows() << '\n';
    output << prefix << ".cols=" << matrix.cols() << '\n';
    for (unsigned int row = 0; row < matrix.rows(); ++row)
    {
        for (unsigned int column = 0; column < matrix.cols(); ++column)
        {
            write_interval(
                output,
                prefix + ".entry." + unsigned_text(row) + "." + unsigned_text(column),
                matrix[row][column]);
        }
    }
}

Matrix<Interval> take_matrix_interval(map<string, string> &fields, const string &prefix)
{
    const unsigned long rows = parse_unsigned(take(fields, prefix + ".rows"), prefix);
    const unsigned long cols = parse_unsigned(take(fields, prefix + ".cols"), prefix);
    if (rows == 0 || cols == 0 || rows > 1000 || cols > 1000)
    {
        throw runtime_error("invalid interval matrix dimensions: " + prefix);
    }
    Matrix<Interval> matrix(static_cast<int>(rows), static_cast<int>(cols));
    for (unsigned long row = 0; row < rows; ++row)
    {
        for (unsigned long column = 0; column < cols; ++column)
        {
            matrix[row][column] = take_interval(
                fields,
                prefix + ".entry." + unsigned_text(row) + "." + unsigned_text(column));
        }
    }
    return matrix;
}

void write_matrix_real(ostream &output, const string &prefix, const Matrix<Real> &matrix)
{
    output << prefix << ".rows=" << matrix.rows() << '\n';
    output << prefix << ".cols=" << matrix.cols() << '\n';
    for (unsigned int row = 0; row < matrix.rows(); ++row)
    {
        for (unsigned int column = 0; column < matrix.cols(); ++column)
        {
            write_real(
                output,
                prefix + ".entry." + unsigned_text(row) + "." + unsigned_text(column),
                matrix[row][column]);
        }
    }
}

Matrix<Real> take_matrix_real(map<string, string> &fields, const string &prefix)
{
    const unsigned long rows = parse_unsigned(take(fields, prefix + ".rows"), prefix);
    const unsigned long cols = parse_unsigned(take(fields, prefix + ".cols"), prefix);
    if (rows == 0 || cols == 0 || rows > 1000 || cols > 1000)
    {
        throw runtime_error("invalid real matrix dimensions: " + prefix);
    }
    Matrix<Real> matrix(static_cast<int>(rows), static_cast<int>(cols));
    for (unsigned long row = 0; row < rows; ++row)
    {
        for (unsigned long column = 0; column < cols; ++column)
        {
            matrix[row][column] = parse_real(take(
                fields,
                prefix + ".entry." + unsigned_text(row) + "." + unsigned_text(column)));
        }
    }
    return matrix;
}

void write_settings(ostream &output, const Computational_Setting &setting)
{
    output << "settings.order=" << setting.tm_setting.order << '\n';
    output << "settings.term_ordering=flowstar_polynomial_list_order\n";
    write_interval(output, "settings.cutoff", setting.tm_setting.cutoff_threshold);
    output << "settings.remainder_target_count=" << setting.tm_setting.remainder_estimation.size() << '\n';
    for (size_t index = 0; index < setting.tm_setting.remainder_estimation.size(); ++index)
    {
        write_interval(output, "settings.remainder_target." + unsigned_text(index), setting.tm_setting.remainder_estimation[index]);
    }
    output << "settings.step_exp_count=" << setting.tm_setting.step_exp_table.size() << '\n';
    for (size_t index = 0; index < setting.tm_setting.step_exp_table.size(); ++index)
    {
        write_interval(output, "settings.step_exp." + unsigned_text(index), setting.tm_setting.step_exp_table[index]);
    }
    output << "settings.step_end_exp_count=" << setting.tm_setting.step_end_exp_table.size() << '\n';
    for (size_t index = 0; index < setting.tm_setting.step_end_exp_table.size(); ++index)
    {
        write_real(output, "settings.step_end_exp." + unsigned_text(index), setting.tm_setting.step_end_exp_table[index]);
    }
    write_real(output, "settings.local_step", Real(kStep));
}

void consume_expected_settings(map<string, string> &fields, const Computational_Setting &setting)
{
    ostringstream expected_stream;
    write_settings(expected_stream, setting);
    istringstream input(expected_stream.str());
    string line;
    while (getline(input, line))
    {
        const size_t separator = line.find('=');
        const string key = line.substr(0, separator);
        const string expected = line.substr(separator + 1);
        const string actual = take(fields, key);
        if (actual != expected)
        {
            throw runtime_error("frozen setting mismatch: " + key);
        }
    }
}

void serialize_snapshot(
    ostream &output,
    const Snapshot &snapshot,
    const Computational_Setting &setting)
{
    unsigned long variable_dimension = 0;
    const TaylorModelVec<Real> *vectors[2] = {&snapshot.flowpipe.tmvPre, &snapshot.flowpipe.tmv};
    for (int vector_index = 0; vector_index < 2 && variable_dimension == 0; ++vector_index)
    {
        for (size_t component = 0; component < vectors[vector_index]->tms.size(); ++component)
        {
            const list<Term<Real> > &terms = vectors[vector_index]->tms[component].expansion.terms;
            if (!terms.empty())
            {
                variable_dimension = terms.begin()->dimension();
                break;
            }
        }
    }
    if (variable_dimension == 0)
    {
        throw runtime_error("cannot infer Taylor-model variable dimension");
    }
    output << "schema=" << kSchema << '\n';
    output << "producer=" << snapshot.producer << '\n';
    output << "phase=" << snapshot.phase << '\n';
    output << "step=" << snapshot.step << '\n';
    write_real(output, "local_time", snapshot.local_time);
    output << "state_dimension=" << snapshot.flowpipe.tmvPre.tms.size() << '\n';
    output << "variable_dimension=" << variable_dimension << '\n';
    write_settings(output, setting);
    output << "flowpipe.safety=" << snapshot.flowpipe.safety << '\n';
    output << "flowpipe.constrained=" << (snapshot.flowpipe.bConstrained ? 1 : 0) << '\n';
    output << "flowpipe.domain_count=" << snapshot.flowpipe.domain.size() << '\n';
    for (size_t index = 0; index < snapshot.flowpipe.domain.size(); ++index)
    {
        write_interval(output, "flowpipe.domain." + unsigned_text(index), snapshot.flowpipe.domain[index]);
    }
    write_tmv(output, "flowpipe.tmvPre", snapshot.flowpipe.tmvPre);
    write_tmv(output, "flowpipe.tmv", snapshot.flowpipe.tmv);
    output << "queue.max_size=" << snapshot.queue.max_size << '\n';
    output << "queue.scalars_count=" << snapshot.queue.scalars.size() << '\n';
    for (size_t index = 0; index < snapshot.queue.scalars.size(); ++index)
    {
        write_real(output, "queue.scalar." + unsigned_text(index), snapshot.queue.scalars[index]);
    }
    output << "queue.J_count=" << snapshot.queue.J.size() << '\n';
    for (size_t index = 0; index < snapshot.queue.J.size(); ++index)
    {
        write_matrix_interval(output, "queue.J." + unsigned_text(index), snapshot.queue.J[index]);
    }
    output << "queue.Phi_L_count=" << snapshot.queue.Phi_L.size() << '\n';
    for (size_t index = 0; index < snapshot.queue.Phi_L.size(); ++index)
    {
        write_matrix_real(output, "queue.Phi_L." + unsigned_text(index), snapshot.queue.Phi_L[index]);
    }
}

string serialize_snapshot_string(const Snapshot &snapshot, const Computational_Setting &setting)
{
    ostringstream output;
    serialize_snapshot(output, snapshot, setting);
    return output.str();
}

Snapshot parse_snapshot(const string &path, const Computational_Setting &setting)
{
    map<string, string> fields = read_fields(path);
    if (take(fields, "schema") != kSchema)
    {
        throw runtime_error("wrong schema version");
    }
    Snapshot snapshot;
    snapshot.producer = take(fields, "producer");
    if (snapshot.producer != "flowstar_actual_path" && snapshot.producer != "torch_binary64")
    {
        throw runtime_error("unknown producer");
    }
    snapshot.phase = take(fields, "phase");
    if (snapshot.phase != "pre_reset" && snapshot.phase != "post_reset" && snapshot.phase != "torch_state")
    {
        throw runtime_error("unknown queue phase");
    }
    snapshot.step = parse_unsigned(take(fields, "step"), "step");
    snapshot.local_time = parse_real(take(fields, "local_time"));
    const unsigned long state_dimension = parse_unsigned(take(fields, "state_dimension"), "state_dimension");
    const unsigned long variable_dimension = parse_unsigned(take(fields, "variable_dimension"), "variable_dimension");
    if (state_dimension == 0 || variable_dimension == 0 || state_dimension > 1000 || variable_dimension > 1000)
    {
        throw runtime_error("invalid dimensional metadata");
    }
    consume_expected_settings(fields, setting);
    snapshot.flowpipe.safety = static_cast<int>(parse_unsigned(take(fields, "flowpipe.safety"), "flowpipe.safety"));
    const unsigned long constrained = parse_unsigned(take(fields, "flowpipe.constrained"), "flowpipe.constrained");
    if (constrained > 1)
    {
        throw runtime_error("invalid constrained flag");
    }
    snapshot.flowpipe.bConstrained = constrained == 1;
    const unsigned long domain_count = parse_unsigned(take(fields, "flowpipe.domain_count"), "flowpipe.domain_count");
    if (domain_count != variable_dimension)
    {
        throw runtime_error("domain/variable dimension mismatch");
    }
    snapshot.flowpipe.domain.clear();
    for (unsigned long index = 0; index < domain_count; ++index)
    {
        snapshot.flowpipe.domain.push_back(take_interval(fields, "flowpipe.domain." + unsigned_text(index)));
    }
    snapshot.flowpipe.tmvPre = take_tmv(fields, "flowpipe.tmvPre", state_dimension, variable_dimension, kOrder);
    snapshot.flowpipe.tmv = take_tmv(fields, "flowpipe.tmv", state_dimension, variable_dimension, kOrder);
    snapshot.queue.max_size = static_cast<unsigned int>(parse_unsigned(take(fields, "queue.max_size"), "queue.max_size"));
    if (snapshot.queue.max_size == 0)
    {
        throw runtime_error("queue max_size must be positive");
    }
    const unsigned long scalar_count = parse_unsigned(take(fields, "queue.scalars_count"), "queue.scalars_count");
    if (scalar_count != state_dimension)
    {
        throw runtime_error("queue scalar/state dimension mismatch");
    }
    snapshot.queue.scalars.clear();
    for (unsigned long index = 0; index < scalar_count; ++index)
    {
        snapshot.queue.scalars.push_back(parse_real(take(fields, "queue.scalar." + unsigned_text(index))));
    }
    const unsigned long j_count = parse_unsigned(take(fields, "queue.J_count"), "queue.J_count");
    if (j_count > snapshot.queue.max_size)
    {
        throw runtime_error("queue J exceeds max_size");
    }
    snapshot.queue.J.clear();
    for (unsigned long index = 0; index < j_count; ++index)
    {
        Matrix<Interval> matrix = take_matrix_interval(fields, "queue.J." + unsigned_text(index));
        if (matrix.rows() != state_dimension || matrix.cols() != 1)
        {
            throw runtime_error("wrong J dimensions");
        }
        snapshot.queue.J.push_back(matrix);
    }
    const unsigned long phi_count = parse_unsigned(take(fields, "queue.Phi_L_count"), "queue.Phi_L_count");
    if (phi_count != j_count)
    {
        throw runtime_error("Phi_L/J count mismatch");
    }
    snapshot.queue.Phi_L.clear();
    for (unsigned long index = 0; index < phi_count; ++index)
    {
        Matrix<Real> matrix = take_matrix_real(fields, "queue.Phi_L." + unsigned_text(index));
        if (matrix.rows() != state_dimension || matrix.cols() != state_dimension)
        {
            throw runtime_error("wrong Phi_L dimensions");
        }
        snapshot.queue.Phi_L.push_back(matrix);
    }
    if (!fields.empty())
    {
        throw runtime_error("unknown/extra schema field: " + fields.begin()->first);
    }
    return snapshot;
}

void write_text(const string &path, const string &contents)
{
    ofstream output(path.c_str(), ios::binary);
    if (!output)
    {
        throw runtime_error("cannot write: " + path);
    }
    output << contents;
}

string read_text(const string &path)
{
    ifstream input(path.c_str(), ios::binary);
    if (!input)
    {
        throw runtime_error("cannot read: " + path);
    }
    ostringstream output;
    output << input.rdbuf();
    return output.str();
}

void configure(
    Variables &variables,
    ODE<Real> *&ode,
    Computational_Setting *&setting,
    Flowpipe *&initial)
{
    const int x_id = variables.declareVar("x");
    const int y_id = variables.declareVar("y");
    variables.declareVar("t");
    ode = new ODE<Real>({"y", "(1 - x^2) * y - x", "1"}, variables);
    setting = new Computational_Setting(variables);
    setting->printOff();
    if (!setting->setFixedStepsize(kStep, kOrder))
    {
        throw runtime_error("cannot configure fixed Flow* setting");
    }
    setting->setCutoffThreshold(1e-10);
    vector<Interval> target(ode->expressions.size(), Interval(-1e-4, 1e-4));
    setting->setRemainderEstimation(target);
    vector<Interval> box(variables.size());
    // Frozen 2026-08-15 lane: MPFR parses the decimal endpoints with directed
    // rounding before Flowpipe's affine normalization.
    box[x_id] = Interval("1.1", "1.4");
    box[y_id] = Interval("2.35", "2.45");
    initial = new Flowpipe(box);
}

Computational_Setting *observer_setting = NULL;
string observer_directory;
vector<Snapshot> captured;
const set<unsigned long> selected_steps = {
    1, 2, 10, 50, 99, 100, 101, 200, 299, 300, 397, 474, 631, 632
};

string fixture_name(const unsigned long step, const string &phase)
{
    return observer_directory + "/step_" + unsigned_text(step) + "_" + phase + ".state";
}

void record_snapshot(
    const Flowpipe &flowpipe,
    const Symbolic_Remainder &queue,
    const unsigned long step,
    const char *phase)
{
    if (selected_steps.find(step) == selected_steps.end())
    {
        return;
    }
    Snapshot snapshot;
    snapshot.producer = "flowstar_actual_path";
    snapshot.phase = phase;
    snapshot.step = step;
    snapshot.local_time = Real(static_cast<double>(step) * kStep);
    snapshot.flowpipe = flowpipe;
    snapshot.queue = queue;
    const string path = fixture_name(step, phase);
    write_text(path, serialize_snapshot_string(snapshot, *observer_setting));
    captured.push_back(snapshot);
}

string base_name(const string &path)
{
    const size_t position = path.find_last_of('/');
    return position == string::npos ? path : path.substr(position + 1);
}

int export_and_verify(const string &directory)
{
    if (mkdir(directory.c_str(), 0775) != 0 && errno != EEXIST)
    {
        throw runtime_error("cannot create fixture directory");
    }
    Variables variables;
    ODE<Real> *ode = NULL;
    Computational_Setting *setting = NULL;
    Flowpipe *initial = NULL;
    configure(variables, ode, setting, initial);
    observer_setting = setting;
    observer_directory = directory;
    captured.clear();
    Symbolic_Remainder queue(*initial, 100);
    Result_of_Reachability result;
    vector<Constraint> safe_set;
    ode->reach(result, *initial, 10.0, *setting, safe_set, queue);
    if (!result.isCompleted() || result.flowpipes.size() != 1000)
    {
        throw runtime_error("actual-path fixture export did not complete 1000 steps");
    }

    unsigned long roundtrip_count = 0;
    unsigned long continuation_count = 0;
    for (size_t index = 0; index < captured.size(); ++index)
    {
        const Snapshot &original = captured[index];
        const string path = fixture_name(original.step, original.phase);
        Snapshot imported = parse_snapshot(path, *setting);
        const string reencoded = serialize_snapshot_string(imported, *setting);
        if (read_text(path) != reencoded)
        {
            throw runtime_error("canonical byte roundtrip mismatch: " + base_name(path));
        }
        write_text(path + ".roundtrip", reencoded);
        ++roundtrip_count;

        Snapshot original_next = original;
        Snapshot imported_next = imported;
        if (original.phase == "pre_reset" && original_next.queue.J.size() >= original_next.queue.max_size)
        {
            original_next.queue.reset(original_next.flowpipe.tmvPre.tms.size());
            imported_next.queue.reset(imported_next.flowpipe.tmvPre.tms.size());
        }
        Flowpipe next_a;
        Flowpipe next_b;
        const int result_a = original_next.flowpipe.advance(
            next_a, ode->expressions, setting->tm_setting, vector<Constraint>(), setting->g_setting, original_next.queue);
        const int result_b = imported_next.flowpipe.advance(
            next_b, ode->expressions, setting->tm_setting, vector<Constraint>(), setting->g_setting, imported_next.queue);
        if (result_a != result_b)
        {
            throw runtime_error("roundtrip continuation decision mismatch");
        }
        original_next.flowpipe = next_a;
        imported_next.flowpipe = next_b;
        original_next.phase = imported_next.phase = "pre_reset";
        original_next.step = imported_next.step = original.step + 1;
        original_next.local_time = imported_next.local_time = Real(static_cast<double>(original.step + 1) * kStep);
        if (serialize_snapshot_string(original_next, *setting) != serialize_snapshot_string(imported_next, *setting))
        {
            throw runtime_error("roundtrip continuation state/queue mismatch");
        }
        ++continuation_count;
    }
    ostringstream summary;
    summary << "{\n"
            << "  \"schema\": \"flowstar_lossless_bridge_verification_v1\",\n"
            << "  \"actual_path_accepted_steps\": 1000,\n"
            << "  \"fixture_count\": " << captured.size() << ",\n"
            << "  \"canonical_byte_roundtrips_exact\": " << roundtrip_count << ",\n"
            << "  \"next_step_roundtrips_exact\": " << continuation_count << ",\n"
            << "  \"queue_reset_fixture_steps\": [99, 100, 101],\n"
            << "  \"common_box_reboxing\": false,\n"
            << "  \"decimal_canonicalization\": false,\n"
            << "  \"status\": \"SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE\"\n"
            << "}\n";
    write_text(directory + "/summary.json", summary.str());
    delete initial;
    delete setting;
    delete ode;
    observer_setting = NULL;
    return 0;
}

int selftest(const string &path)
{
    const double values[] = {
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -123456789.25,
        numeric_limits<double>::denorm_min(),
        numeric_limits<double>::min(),
        numeric_limits<double>::max()
    };
    ostringstream output;
    output << "{\n  \"schema\": \"flowstar_mpfr_canonical_selftest_v1\",\n  \"cases\": [\n";
    for (size_t index = 0; index < sizeof(values) / sizeof(values[0]); ++index)
    {
        const Real original(values[index]);
        const string encoded = canonical_real(original);
        const Real decoded = parse_real(encoded);
        if (canonical_real(decoded) != encoded || decoded.toDouble() != values[index])
        {
            throw runtime_error("MPFR canonical selftest mismatch");
        }
        output << "    {\"index\": " << index << ", \"canonical\": \"" << encoded
               << "\", \"float_hex\": \"";
        ostringstream hex;
        hex << hexfloat << values[index];
        output << hex.str() << "\"}";
        if (index + 1 < sizeof(values) / sizeof(values[0]))
        {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n  \"all_exact\": true\n}\n";
    write_text(path, output.str());
    return 0;
}

int continue_snapshot(const string &input_path, const string &output_path)
{
    Variables variables;
    ODE<Real> *ode = NULL;
    Computational_Setting *setting = NULL;
    Flowpipe *initial = NULL;
    configure(variables, ode, setting, initial);
    Snapshot snapshot = parse_snapshot(input_path, *setting);
    const size_t ode_dimension = ode->expressions.size();
    if (snapshot.flowpipe.tmvPre.tms.size() != ode_dimension
        || snapshot.flowpipe.tmv.tms.size() != ode_dimension
        || snapshot.queue.scalars.size() != ode_dimension)
    {
        delete initial;
        delete setting;
        delete ode;
        throw runtime_error(
            "schema/operator mismatch: frozen Flow* VDP operator requires three "
            "state components (x,y,t) and a three-component symbolic queue");
    }
    if (snapshot.flowpipe.domain.size() != ode_dimension + 1)
    {
        delete initial;
        delete setting;
        delete ode;
        throw runtime_error(
            "schema/operator mismatch: frozen Flow* VDP operator requires the "
            "local-time plus three normalized-variable domain");
    }
    if (snapshot.phase == "pre_reset" && snapshot.queue.J.size() >= snapshot.queue.max_size)
    {
        snapshot.queue.reset(snapshot.flowpipe.tmvPre.tms.size());
    }
    Flowpipe next;
    const int result = snapshot.flowpipe.advance(
        next,
        ode->expressions,
        setting->tm_setting,
        vector<Constraint>(),
        setting->g_setting,
        snapshot.queue);
    if (result != 1)
    {
        delete initial;
        delete setting;
        delete ode;
        throw runtime_error("Flow* continuation did not accept the requested next step");
    }
    snapshot.flowpipe = next;
    snapshot.phase = "pre_reset";
    ++snapshot.step;
    snapshot.local_time = Real(static_cast<double>(snapshot.step) * kStep);
    write_text(output_path, serialize_snapshot_string(snapshot, *setting));
    delete initial;
    delete setting;
    delete ode;
    return 0;
}

} // namespace

void flowstar::flowstar_causal_observe_reach_step(
    const Flowpipe &flowpipe,
    const Symbolic_Remainder &queue,
    const unsigned long accepted_step,
    const double,
    const double,
    const char *phase)
{
    record_snapshot(flowpipe, queue, accepted_step, phase);
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 3 && string(argv[1]) == "export-fixtures")
        {
            return export_and_verify(argv[2]);
        }
        if (argc == 4 && string(argv[1]) == "roundtrip")
        {
            Variables variables;
            ODE<Real> *ode = NULL;
            Computational_Setting *setting = NULL;
            Flowpipe *initial = NULL;
            configure(variables, ode, setting, initial);
            const Snapshot snapshot = parse_snapshot(argv[2], *setting);
            write_text(argv[3], serialize_snapshot_string(snapshot, *setting));
            delete initial;
            delete setting;
            delete ode;
            return read_text(argv[2]) == read_text(argv[3]) ? 0 : 1;
        }
        if (argc == 3 && string(argv[1]) == "validate")
        {
            Variables variables;
            ODE<Real> *ode = NULL;
            Computational_Setting *setting = NULL;
            Flowpipe *initial = NULL;
            configure(variables, ode, setting, initial);
            parse_snapshot(argv[2], *setting);
            delete initial;
            delete setting;
            delete ode;
            return 0;
        }
        if (argc == 3 && string(argv[1]) == "selftest")
        {
            return selftest(argv[2]);
        }
        if (argc == 4 && string(argv[1]) == "continue")
        {
            return continue_snapshot(argv[2], argv[3]);
        }
        cerr << "usage: flowstar_lossless_state_queue_bridge "
             << "export-fixtures DIR | roundtrip INPUT OUTPUT | validate INPUT | "
             << "selftest OUTPUT | continue INPUT OUTPUT\n";
        return 2;
    }
    catch (const exception &error)
    {
        cerr << error.what() << '\n';
        return 3;
    }
}
