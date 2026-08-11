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

namespace
{

const mpfr_prec_t kPrecision = 256;

struct Interval
{
    mpfr_t lo;
    mpfr_t hi;

    Interval()
    {
        mpfr_init2(lo, kPrecision);
        mpfr_init2(hi, kPrecision);
        mpfr_set_zero(lo, 0);
        mpfr_set_zero(hi, 0);
    }

    Interval(const Interval &other)
    {
        mpfr_init2(lo, kPrecision);
        mpfr_init2(hi, kPrecision);
        mpfr_set(lo, other.lo, MPFR_RNDN);
        mpfr_set(hi, other.hi, MPFR_RNDN);
    }

    Interval &operator=(const Interval &other)
    {
        if (this != &other)
        {
            mpfr_set(lo, other.lo, MPFR_RNDN);
            mpfr_set(hi, other.hi, MPFR_RNDN);
        }
        return *this;
    }

    ~Interval()
    {
        mpfr_clear(lo);
        mpfr_clear(hi);
    }
};

void set_binary64(Interval &out, const std::string &lo_hex, const std::string &hi_hex)
{
    char *lo_end = NULL;
    char *hi_end = NULL;
    const double lo = std::strtod(lo_hex.c_str(), &lo_end);
    const double hi = std::strtod(hi_hex.c_str(), &hi_end);
    if (lo_end == lo_hex.c_str() || *lo_end != '\0' || hi_end == hi_hex.c_str() || *hi_end != '\0' || lo > hi)
    {
        throw std::runtime_error("invalid binary64 interval input");
    }
    mpfr_set_d(out.lo, lo, MPFR_RNDN);
    mpfr_set_d(out.hi, hi, MPFR_RNDN);
}

Interval add(const Interval &left, const Interval &right)
{
    Interval out;
    mpfr_add(out.lo, left.lo, right.lo, MPFR_RNDD);
    mpfr_add(out.hi, left.hi, right.hi, MPFR_RNDU);
    return out;
}

Interval multiply(const Interval &left, const Interval &right)
{
    Interval out;
    mpfr_t candidate_lo;
    mpfr_t candidate_hi;
    mpfr_init2(candidate_lo, kPrecision);
    mpfr_init2(candidate_hi, kPrecision);
    bool first = true;
    const mpfr_srcptr left_values[] = {left.lo, left.hi};
    const mpfr_srcptr right_values[] = {right.lo, right.hi};
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            mpfr_mul(candidate_lo, left_values[i], right_values[j], MPFR_RNDD);
            mpfr_mul(candidate_hi, left_values[i], right_values[j], MPFR_RNDU);
            if (first || mpfr_less_p(candidate_lo, out.lo))
            {
                mpfr_set(out.lo, candidate_lo, MPFR_RNDN);
            }
            if (first || mpfr_greater_p(candidate_hi, out.hi))
            {
                mpfr_set(out.hi, candidate_hi, MPFR_RNDN);
            }
            first = false;
        }
    }
    mpfr_clear(candidate_lo);
    mpfr_clear(candidate_hi);
    return out;
}

Interval negate(const Interval &value)
{
    Interval out;
    mpfr_neg(out.lo, value.hi, MPFR_RNDD);
    mpfr_neg(out.hi, value.lo, MPFR_RNDU);
    return out;
}

void print_result(const std::string &node, const Interval &value)
{
    const double lo = mpfr_get_d(value.lo, MPFR_RNDD);
    const double hi = mpfr_get_d(value.hi, MPFR_RNDU);
    std::cout << "ORACLE_RESULT node=" << node
              << " precision_bits=" << kPrecision
              << " input_semantics=exact_binary64"
              << " rounding=mpfr_directed"
              << " lo_decimal=" << std::setprecision(17) << lo
              << " lo_hex=" << std::hexfloat << lo << std::defaultfloat
              << " hi_decimal=" << std::setprecision(17) << hi
              << " hi_hex=" << std::hexfloat << hi << std::defaultfloat
              << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " interval_dag.tsv\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    if (!input)
    {
        std::cerr << "unable to open input\n";
        return 3;
    }
    std::map<std::string, Interval> nodes;
    std::vector<std::string> emissions;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream parser(line);
        std::string operation;
        std::string name;
        if (!(parser >> operation >> name))
        {
            std::cerr << "invalid input row: " << line << "\n";
            return 4;
        }
        if (operation == "literal")
        {
            std::string lo_hex;
            std::string hi_hex;
            if (!(parser >> lo_hex >> hi_hex))
            {
                std::cerr << "invalid literal row: " << line << "\n";
                return 5;
            }
            Interval value;
            try
            {
                set_binary64(value, lo_hex, hi_hex);
            }
            catch (const std::exception &error)
            {
                std::cerr << error.what() << ": " << line << "\n";
                return 5;
            }
            nodes[name] = value;
        }
        else if (operation == "add" || operation == "mul")
        {
            std::string left;
            std::string right;
            if (!(parser >> left >> right) || nodes.find(left) == nodes.end() || nodes.find(right) == nodes.end())
            {
                std::cerr << "invalid or non-prior binary parents: " << line << "\n";
                return 6;
            }
            nodes[name] = operation == "add" ? add(nodes[left], nodes[right]) : multiply(nodes[left], nodes[right]);
        }
        else if (operation == "neg")
        {
            std::string parent;
            if (!(parser >> parent) || nodes.find(parent) == nodes.end())
            {
                std::cerr << "invalid or non-prior negate parent: " << line << "\n";
                return 7;
            }
            nodes[name] = negate(nodes[parent]);
        }
        else if (operation == "emit")
        {
            if (nodes.find(name) == nodes.end())
            {
                std::cerr << "unknown emit node: " << line << "\n";
                return 8;
            }
            emissions.push_back(name);
        }
        else
        {
            std::cerr << "unknown operation: " << line << "\n";
            return 9;
        }
    }
    for (std::vector<std::string>::const_iterator name = emissions.begin(); name != emissions.end(); ++name)
    {
        print_result(*name, nodes[*name]);
    }
    return 0;
}
