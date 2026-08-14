// Independent MPFR natural-interval oracle for the frozen VDP step-1 input.
#include <gmp.h>
#include <mpfr.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Interval {
    mpfr_t lo;
    mpfr_t hi;
    mpfr_prec_t precision;

    explicit Interval(mpfr_prec_t p) : precision(p) {
        mpfr_init2(lo, p);
        mpfr_init2(hi, p);
        mpfr_set_zero(lo, 0);
        mpfr_set_zero(hi, 0);
    }

    Interval(const Interval &other) : precision(other.precision) {
        mpfr_init2(lo, precision);
        mpfr_init2(hi, precision);
        mpfr_set(lo, other.lo, MPFR_RNDN);
        mpfr_set(hi, other.hi, MPFR_RNDN);
    }

    Interval &operator=(const Interval &other) {
        if (this == &other) return *this;
        if (precision != other.precision) {
            mpfr_set_prec(lo, other.precision);
            mpfr_set_prec(hi, other.precision);
            precision = other.precision;
        }
        mpfr_set(lo, other.lo, MPFR_RNDN);
        mpfr_set(hi, other.hi, MPFR_RNDN);
        return *this;
    }

    ~Interval() {
        mpfr_clear(lo);
        mpfr_clear(hi);
    }
};

struct Rational {
    string numerator;
    string denominator;
};

struct Domain {
    Rational lo;
    Rational hi;
};

struct Term {
    unsigned int exponent[3];
    Rational coefficient;
};

typedef vector<Term> Polynomial;

void set_rational(mpfr_t output, const Rational &value, mpfr_rnd_t rounding) {
    mpq_t rational;
    mpq_init(rational);
    if (mpz_set_str(mpq_numref(rational), value.numerator.c_str(), 10) != 0 ||
        mpz_set_str(mpq_denref(rational), value.denominator.c_str(), 10) != 0 ||
        mpz_sgn(mpq_denref(rational)) == 0) {
        mpq_clear(rational);
        throw runtime_error("invalid rational input");
    }
    mpq_canonicalize(rational);
    mpfr_set_q(output, rational, rounding);
    mpq_clear(rational);
}

Interval rational_interval(const Domain &domain, mpfr_prec_t precision) {
    Interval result(precision);
    set_rational(result.lo, domain.lo, MPFR_RNDD);
    set_rational(result.hi, domain.hi, MPFR_RNDU);
    if (mpfr_greater_p(result.lo, result.hi)) throw runtime_error("reversed domain");
    return result;
}

Interval rational_point(const Rational &value, mpfr_prec_t precision) {
    Domain point;
    point.lo = value;
    point.hi = value;
    return rational_interval(point, precision);
}

Interval integer_point(long value, mpfr_prec_t precision) {
    Interval result(precision);
    mpfr_set_si(result.lo, value, MPFR_RNDD);
    mpfr_set_si(result.hi, value, MPFR_RNDU);
    return result;
}

Interval add(const Interval &left, const Interval &right) {
    Interval result(left.precision);
    mpfr_add(result.lo, left.lo, right.lo, MPFR_RNDD);
    mpfr_add(result.hi, left.hi, right.hi, MPFR_RNDU);
    return result;
}

Interval negate(const Interval &value) {
    Interval result(value.precision);
    mpfr_neg(result.lo, value.hi, MPFR_RNDD);
    mpfr_neg(result.hi, value.lo, MPFR_RNDU);
    return result;
}

Interval subtract(const Interval &left, const Interval &right) {
    return add(left, negate(right));
}

Interval multiply(const Interval &left, const Interval &right) {
    Interval result(left.precision);
    mpfr_t candidate;
    mpfr_init2(candidate, left.precision);
    bool first = true;
    const mpfr_srcptr left_values[2] = {left.lo, left.hi};
    const mpfr_srcptr right_values[2] = {right.lo, right.hi};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            mpfr_mul(candidate, left_values[i], right_values[j], MPFR_RNDD);
            if (first || mpfr_less_p(candidate, result.lo)) mpfr_set(result.lo, candidate, MPFR_RNDN);
            first = false;
        }
    }
    first = true;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            mpfr_mul(candidate, left_values[i], right_values[j], MPFR_RNDU);
            if (first || mpfr_greater_p(candidate, result.hi)) mpfr_set(result.hi, candidate, MPFR_RNDN);
            first = false;
        }
    }
    mpfr_clear(candidate);
    return result;
}

Interval power(const Interval &value, unsigned int exponent) {
    if (exponent == 0) return integer_point(1, value.precision);
    Interval result(value.precision);
    if (exponent % 2 == 1) {
        mpfr_pow_ui(result.lo, value.lo, exponent, MPFR_RNDD);
        mpfr_pow_ui(result.hi, value.hi, exponent, MPFR_RNDU);
        return result;
    }
    const bool crosses_zero = mpfr_sgn(value.lo) <= 0 && mpfr_sgn(value.hi) >= 0;
    mpfr_t left_down, right_down, left_up, right_up;
    mpfr_inits2(value.precision, left_down, right_down, left_up, right_up, (mpfr_ptr) 0);
    mpfr_pow_ui(left_down, value.lo, exponent, MPFR_RNDD);
    mpfr_pow_ui(right_down, value.hi, exponent, MPFR_RNDD);
    mpfr_pow_ui(left_up, value.lo, exponent, MPFR_RNDU);
    mpfr_pow_ui(right_up, value.hi, exponent, MPFR_RNDU);
    if (crosses_zero) mpfr_set_zero(result.lo, 0);
    else mpfr_set(result.lo, mpfr_less_p(left_down, right_down) ? left_down : right_down, MPFR_RNDN);
    mpfr_set(result.hi, mpfr_greater_p(left_up, right_up) ? left_up : right_up, MPFR_RNDN);
    mpfr_clears(left_down, right_down, left_up, right_up, (mpfr_ptr) 0);
    return result;
}

Interval polynomial_range(const Polynomial &polynomial, const vector<Interval> &domain, mpfr_prec_t precision) {
    Interval result = integer_point(0, precision);
    for (Polynomial::const_iterator term = polynomial.begin(); term != polynomial.end(); ++term) {
        Interval contribution = rational_point(term->coefficient, precision);
        for (unsigned int variable = 0; variable < 3; ++variable) {
            contribution = multiply(contribution, power(domain[variable], term->exponent[variable]));
        }
        result = add(result, contribution);
    }
    return result;
}

string canonical(const mpfr_t value, mpfr_prec_t precision) {
    mpz_t significand;
    mpz_init(significand);
    const mpfr_exp_t exponent = mpfr_get_z_2exp(significand, value);
    const int sign = mpz_sgn(significand) < 0 ? -1 : 1;
    if (mpz_sgn(significand) < 0) mpz_neg(significand, significand);
    char *digits = mpz_get_str(NULL, 16, significand);
    ostringstream output;
    output << precision << ':' << sign << ':' << digits << ':' << exponent;
    free(digits);
    mpz_clear(significand);
    return output.str();
}

string decimal(const mpfr_t value, mpfr_prec_t precision) {
    const int digits = static_cast<int>(precision * 0.30103) + 8;
    char *text = NULL;
    mpfr_asprintf(&text, "%.*Re", digits, value);
    const string result(text);
    mpfr_free_str(text);
    return result;
}

void emit_interval(ostream &output, const Interval &value) {
    output << "{\"lower\":{\"decimal\":\"" << decimal(value.lo, value.precision)
           << "\",\"canonical_mpfr\":\"" << canonical(value.lo, value.precision)
           << "\"},\"upper\":{\"decimal\":\"" << decimal(value.hi, value.precision)
           << "\",\"canonical_mpfr\":\"" << canonical(value.hi, value.precision)
           << "\"},\"precision_bits\":" << value.precision
           << ",\"rounding\":\"lower=MPFR_RNDD;upper=MPFR_RNDU\"}";
}

bool subset(const Interval &inner, const Interval &outer) {
    return mpfr_greaterequal_p(inner.lo, outer.lo) && mpfr_lessequal_p(inner.hi, outer.hi);
}

Interval margin(const Interval &inner, const Interval &outer) {
    Interval result(inner.precision);
    mpfr_t left, right;
    mpfr_inits2(inner.precision, left, right, (mpfr_ptr) 0);
    mpfr_sub(left, inner.lo, outer.lo, MPFR_RNDD);
    mpfr_sub(right, outer.hi, inner.hi, MPFR_RNDD);
    mpfr_set(result.lo, mpfr_less_p(left, right) ? left : right, MPFR_RNDN);
    mpfr_set(result.hi, result.lo, MPFR_RNDN);
    mpfr_clears(left, right, (mpfr_ptr) 0);
    return result;
}

void load_input(
    const string &path,
    map<string, Domain> &domains,
    map<string, Polynomial> &polynomials,
    int &refinement_steps) {
    ifstream input(path.c_str());
    if (!input) throw runtime_error("cannot open oracle input");
    string kind;
    while (input >> kind) {
        if (kind == "domain") {
            string name;
            Domain value;
            input >> name >> value.lo.numerator >> value.lo.denominator
                  >> value.hi.numerator >> value.hi.denominator;
            domains[name] = value;
        } else if (kind == "term") {
            string name;
            Term term;
            input >> name >> term.exponent[0] >> term.exponent[1] >> term.exponent[2]
                  >> term.coefficient.numerator >> term.coefficient.denominator;
            polynomials[name].push_back(term);
        } else if (kind == "poly") {
            string name;
            input >> name;
            polynomials[name] = Polynomial();
        } else if (kind == "refinement_steps") {
            input >> refinement_steps;
        } else if (kind[0] == '#') {
            string remainder;
            getline(input, remainder);
        } else {
            throw runtime_error(string("unknown oracle input record: ") + kind);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << "usage: step1_mpfr_oracle INPUT_TSV PRECISION OUTPUT_JSON\n";
        return 2;
    }
    try {
        const long requested = strtol(argv[2], NULL, 10);
        if (requested < 128) throw runtime_error("precision must be at least 128 bits");
        const mpfr_prec_t precision = static_cast<mpfr_prec_t>(requested);
        mpfr_set_default_prec(precision);
        map<string, Domain> domains;
        map<string, Polynomial> polynomials;
        int refinement_steps = 0;
        load_input(argv[1], domains, polynomials, refinement_steps);
        const char *required_domains[] = {"tau_segment", "ux", "uy", "target"};
        for (size_t i = 0; i < 4; ++i) {
            if (!domains.count(required_domains[i])) throw runtime_error("missing required domain");
        }
        const char *required_polynomials[] = {
            "px", "py", "endpoint_px", "endpoint_py", "residual_x", "residual_y", "truncation_x", "truncation_y"
        };
        for (size_t i = 0; i < 8; ++i) {
            if (!polynomials.count(required_polynomials[i])) throw runtime_error("missing required polynomial");
        }
        vector<Interval> segment_domain;
        segment_domain.push_back(rational_interval(domains["tau_segment"], precision));
        segment_domain.push_back(rational_interval(domains["ux"], precision));
        segment_domain.push_back(rational_interval(domains["uy"], precision));
        vector<Interval> endpoint_domain;
        endpoint_domain.push_back(rational_interval(domains["tau_segment"], precision));
        endpoint_domain.push_back(segment_domain[1]);
        endpoint_domain.push_back(segment_domain[2]);

        const Interval px = polynomial_range(polynomials["px"], segment_domain, precision);
        const Interval py = polynomial_range(polynomials["py"], segment_domain, precision);
        const Interval endpoint_px = polynomial_range(polynomials["endpoint_px"], endpoint_domain, precision);
        const Interval endpoint_py = polynomial_range(polynomials["endpoint_py"], endpoint_domain, precision);
        const Interval residual_x = polynomial_range(polynomials["residual_x"], segment_domain, precision);
        const Interval residual_y = polynomial_range(polynomials["residual_y"], segment_domain, precision);
        const Interval truncation_x = polynomial_range(polynomials["truncation_x"], segment_domain, precision);
        const Interval truncation_y = polynomial_range(polynomials["truncation_y"], segment_domain, precision);
        const Interval zero = integer_point(0, precision);
        const Interval tau = rational_interval(domains["tau_segment"], precision);
        Interval rx = rational_interval(domains["target"], precision);
        Interval ry = rx;
        vector<Interval> image_x_rows;
        vector<Interval> image_y_rows;
        vector<Interval> margin_x_rows;
        vector<Interval> margin_y_rows;
        vector<bool> subset_x_rows;
        vector<bool> subset_y_rows;
        for (int iteration = 0; iteration < refinement_steps; ++iteration) {
            const Interval two = integer_point(2, precision);
            Interval nonlinear = multiply(multiply(px, px), ry);
            nonlinear = add(nonlinear, multiply(multiply(multiply(two, px), rx), py));
            nonlinear = add(nonlinear, multiply(multiply(multiply(two, px), rx), ry));
            nonlinear = add(nonlinear, multiply(multiply(rx, rx), py));
            nonlinear = add(nonlinear, multiply(multiply(rx, rx), ry));
            const Interval derivative_x = ry;
            const Interval derivative_y = subtract(subtract(ry, rx), nonlinear);
            const Interval image_x = add(add(add(residual_x, truncation_x), zero), multiply(derivative_x, tau));
            const Interval image_y = add(add(add(residual_y, truncation_y), zero), multiply(derivative_y, tau));
            image_x_rows.push_back(image_x);
            image_y_rows.push_back(image_y);
            margin_x_rows.push_back(margin(image_x, rx));
            margin_y_rows.push_back(margin(image_y, ry));
            subset_x_rows.push_back(subset(image_x, rx));
            subset_y_rows.push_back(subset(image_y, ry));
            if (!subset_x_rows.back() || !subset_y_rows.back()) break;
            rx = image_x;
            ry = image_y;
        }
        const Interval segment_final_x = add(px, rx);
        const Interval segment_final_y = add(py, ry);
        const Interval endpoint_final_x = add(endpoint_px, rx);
        const Interval endpoint_final_y = add(endpoint_py, ry);

        ofstream output(argv[3]);
        if (!output) throw runtime_error("cannot open output JSON");
        output << "{\n  \"schema\":\"independent_mpfr_step1_oracle_v1\",\n"
               << "  \"precision_bits\":" << precision << ",\n"
               << "  \"range_algorithm\":\"natural_interval_termwise\",\n"
               << "  \"rounding_contract\":\"every lower op MPFR_RNDD; every upper op MPFR_RNDU\",\n"
               << "  \"segment_polynomial\":{\"x\":";
        emit_interval(output, px); output << ",\"y\":"; emit_interval(output, py); output << "},\n";
        output << "  \"endpoint_polynomial\":{\"x\":";
        emit_interval(output, endpoint_px); output << ",\"y\":"; emit_interval(output, endpoint_py); output << "},\n";
        output << "  \"truncation_remainder\":{\"x\":";
        emit_interval(output, truncation_x); output << ",\"y\":"; emit_interval(output, truncation_y); output << "},\n";
        output << "  \"cutoff_remainder\":{\"x\":";
        emit_interval(output, zero); output << ",\"y\":"; emit_interval(output, zero); output << "},\n";
        output << "  \"refinement\":[";
        for (size_t i = 0; i < image_x_rows.size(); ++i) {
            if (i) output << ',';
            output << "{\"iteration\":" << i << ",\"image\":{\"x\":";
            emit_interval(output, image_x_rows[i]); output << ",\"y\":"; emit_interval(output, image_y_rows[i]);
            output << "},\"margin\":{\"x\":"; emit_interval(output, margin_x_rows[i]);
            output << ",\"y\":"; emit_interval(output, margin_y_rows[i]);
            output << "},\"subset\":{\"x\":" << (subset_x_rows[i] ? "true" : "false")
                   << ",\"y\":" << (subset_y_rows[i] ? "true" : "false") << "}}";
        }
        output << "],\n  \"final_remainder\":{\"x\":"; emit_interval(output, rx);
        output << ",\"y\":"; emit_interval(output, ry); output << "},\n";
        output << "  \"segment_final\":{\"x\":"; emit_interval(output, segment_final_x);
        output << ",\"y\":"; emit_interval(output, segment_final_y); output << "},\n";
        output << "  \"endpoint_final\":{\"x\":"; emit_interval(output, endpoint_final_x);
        output << ",\"y\":"; emit_interval(output, endpoint_final_y); output << "}\n}\n";
        return 0;
    } catch (const exception &error) {
        cerr << error.what() << '\n';
        return 1;
    }
}
