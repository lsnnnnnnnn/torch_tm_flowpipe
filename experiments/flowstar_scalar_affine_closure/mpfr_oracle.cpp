#include <mpfr.h>

#include <cstdio>
#include <cstdlib>

static void evaluate(
	mpfr_t result, const char * x0_text, const char * time_text,
	const mpfr_rnd_t rounding)
{
	mpfr_t x0;
	mpfr_t time;
	mpfr_t two_time;
	mpfr_t exponential;
	mpfr_t half;
	mpfr_t shifted;
	mpfr_inits2(256, x0, time, two_time, exponential, half, shifted, (mpfr_ptr)0);

	if(mpfr_set_str(x0, x0_text, 10, rounding) != 0 ||
		mpfr_set_str(time, time_text, 10, rounding) != 0)
	{
		fprintf(stderr, "invalid decimal input\n");
		exit(3);
	}
	mpfr_set_d(half, 0.5, MPFR_RNDN);
	mpfr_mul_ui(two_time, time, 2, rounding);
	mpfr_exp(exponential, two_time, rounding);
	mpfr_add(shifted, x0, half, rounding);
	mpfr_mul(result, shifted, exponential, rounding);
	mpfr_sub(result, result, half, rounding);
	mpfr_clears(x0, time, two_time, exponential, half, shifted, (mpfr_ptr)0);
}

static void print_bound(
	const char * name, const mpfr_t value, const mpfr_rnd_t rounding,
	const char * direction)
{
	const double binary64 = mpfr_get_d(value, rounding);
	printf("ORACLE_BOUND name=%s direction=%s decimal=", name, direction);
	mpfr_printf("%.80Re", value);
	printf(" binary64=%.17g binary64_hex=%a\n", binary64, binary64);
}

int main(const int argc, char ** argv)
{
	if(argc != 4)
	{
		fprintf(stderr, "usage: %s x0_lower x0_upper h\n", argv[0]);
		return 2;
	}

	mpfr_t endpoint_lower;
	mpfr_t endpoint_upper;
	mpfr_t tube_lower;
	mpfr_t tube_upper;
	mpfr_inits2(
		256, endpoint_lower, endpoint_upper, tube_lower, tube_upper, (mpfr_ptr)0);
	evaluate(endpoint_lower, argv[1], argv[3], MPFR_RNDD);
	evaluate(endpoint_upper, argv[2], argv[3], MPFR_RNDU);
	evaluate(tube_lower, argv[1], "0", MPFR_RNDD);
	evaluate(tube_upper, argv[2], argv[3], MPFR_RNDU);

	printf(
		"ORACLE_META precision_bits=256 rounding=explicit_directed "
		"formula=((x0+1/2)*exp(2*t))-1/2 "
		"monotone_x0=1 monotone_time=1 "
		"monotone_reason=exp(2*t)>0_and_(1+2*x0)*exp(2*t)>0\n");
	print_bound("endpoint_lower", endpoint_lower, MPFR_RNDD, "down");
	print_bound("endpoint_upper", endpoint_upper, MPFR_RNDU, "up");
	print_bound("tube_lower", tube_lower, MPFR_RNDD, "down");
	print_bound("tube_upper", tube_upper, MPFR_RNDU, "up");

	mpfr_clears(endpoint_lower, endpoint_upper, tube_lower, tube_upper, (mpfr_ptr)0);
	return 0;
}
