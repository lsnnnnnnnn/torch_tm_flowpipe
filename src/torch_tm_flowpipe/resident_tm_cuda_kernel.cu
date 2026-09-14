// This is deliberately a small, fixed production graph for the two-state
// accepted-boundary maps used by the frozen VDP and Brusselator runs.  NVRTC is
// invoked with --fmad=false and --ftz=false.  Every interval endpoint operation
// uses an explicit directed binary64 intrinsic.

#define MAX_ORDER 6
#define MAX_TERMS 28
#define MAX_DROP_TERMS 63
#define MAX_SPLIT 16
#define DIAG_COUNT 9

struct LocalTM {
    double point[MAX_TERMS];
    double lo[MAX_TERMS];
    double hi[MAX_TERMS];
    double rem_lo;
    double rem_hi;
    int split;
};

__device__ __forceinline__ int term_count(int order) {
    return (order + 1) * (order + 2) / 2;
}

// The canonical basis is degree-major and then increasing exponent of u0:
// (0,0), (0,1), (1,0), (0,2), (1,1), (2,0), ...
__device__ __forceinline__ int term_index(int e0, int e1) {
    int degree = e0 + e1;
    return degree * (degree + 1) / 2 + e0;
}

__device__ __forceinline__ bool exact_zero(double lo, double hi) {
    return lo == 0.0 && hi == 0.0;
}

__device__ __forceinline__ void interval_add(
    double a_lo, double a_hi, double b_lo, double b_hi,
    double *out_lo, double *out_hi
) {
    if (exact_zero(b_lo, b_hi)) {
        *out_lo = a_lo;
        *out_hi = a_hi;
    } else if (exact_zero(a_lo, a_hi)) {
        *out_lo = b_lo;
        *out_hi = b_hi;
    } else {
        *out_lo = __dadd_rd(a_lo, b_lo);
        *out_hi = __dadd_ru(a_hi, b_hi);
    }
}

__device__ __forceinline__ void interval_mul(
    double a_lo, double a_hi, double b_lo, double b_hi,
    double *out_lo, double *out_hi
) {
    if (exact_zero(a_lo, a_hi) || exact_zero(b_lo, b_hi)) {
        *out_lo = 0.0;
        *out_hi = 0.0;
        return;
    }
    double ll_lo = __dmul_rd(a_lo, b_lo);
    double lh_lo = __dmul_rd(a_lo, b_hi);
    double hl_lo = __dmul_rd(a_hi, b_lo);
    double hh_lo = __dmul_rd(a_hi, b_hi);
    double ll_hi = __dmul_ru(a_lo, b_lo);
    double lh_hi = __dmul_ru(a_lo, b_hi);
    double hl_hi = __dmul_ru(a_hi, b_lo);
    double hh_hi = __dmul_ru(a_hi, b_hi);
    *out_lo = fmin(fmin(ll_lo, lh_lo), fmin(hl_lo, hh_lo));
    *out_hi = fmax(fmax(ll_hi, lh_hi), fmax(hl_hi, hh_hi));
}

__device__ __forceinline__ double interval_width(double lo, double hi) {
    return lo == hi ? 0.0 : __dsub_ru(hi, lo);
}

__device__ __forceinline__ void zero_tm(LocalTM *value, int terms, int split) {
    for (int index = 0; index < terms; ++index) {
        value->point[index] = 0.0;
        value->lo[index] = 0.0;
        value->hi[index] = 0.0;
    }
    value->rem_lo = 0.0;
    value->rem_hi = 0.0;
    value->split = split;
}

__device__ __forceinline__ double partition_endpoint(
    double lo, double hi, int index, int pieces
) {
    if (index <= 0) return lo;
    if (index >= pieces) return hi;
    double width = hi - lo;
    double fraction = ((double)index) / ((double)pieces);
    double value = lo + width * fraction;
    return fmin(hi, fmax(lo, value));
}

__device__ __forceinline__ void interval_power(
    double base_lo, double base_hi, int exponent,
    double *out_lo, double *out_hi
) {
    double lo = 1.0;
    double hi = 1.0;
    for (int power = 0; power < exponent; ++power) {
        double next_lo, next_hi;
        interval_mul(lo, hi, base_lo, base_hi, &next_lo, &next_hi);
        lo = next_lo;
        hi = next_hi;
    }
    *out_lo = lo;
    *out_hi = hi;
}

__device__ void range_on_box(
    const double *coefficient_lo,
    const double *coefficient_hi,
    int order,
    double d0_lo,
    double d0_hi,
    double d1_lo,
    double d1_hi,
    double *result_lo,
    double *result_hi
) {
    int terms = term_count(order);
    double total_lo = 0.0;
    double total_hi = 0.0;
    for (int degree = 0; degree <= order; ++degree) {
        for (int e0 = 0; e0 <= degree; ++e0) {
            int e1 = degree - e0;
            int index = term_index(e0, e1);
            double c_lo = coefficient_lo[index];
            double c_hi = coefficient_hi[index];
            if (exact_zero(c_lo, c_hi)) continue;
            double p0_lo, p0_hi, p1_lo, p1_hi;
            interval_power(d0_lo, d0_hi, e0, &p0_lo, &p0_hi);
            interval_power(d1_lo, d1_hi, e1, &p1_lo, &p1_hi);
            double mono_lo, mono_hi, term_lo, term_hi;
            interval_mul(p0_lo, p0_hi, p1_lo, p1_hi, &mono_lo, &mono_hi);
            interval_mul(c_lo, c_hi, mono_lo, mono_hi, &term_lo, &term_hi);
            interval_add(total_lo, total_hi, term_lo, term_hi, &total_lo, &total_hi);
        }
    }
    *result_lo = total_lo;
    *result_hi = total_hi;
}

__device__ void range_polynomial(
    const double *coefficient_lo,
    const double *coefficient_hi,
    int order,
    const double *domain_lo,
    const double *domain_hi,
    int split,
    double *result_lo,
    double *result_hi
) {
    int pieces = split > 1 ? split : 1;
    double hull_lo = __longlong_as_double(0x7ff0000000000000ULL);
    double hull_hi = __longlong_as_double(0xfff0000000000000ULL);
    for (int i = 0; i < pieces; ++i) {
        double d0_lo = partition_endpoint(domain_lo[0], domain_hi[0], i, pieces);
        double d0_hi = partition_endpoint(domain_lo[0], domain_hi[0], i + 1, pieces);
        for (int j = 0; j < pieces; ++j) {
            double d1_lo = partition_endpoint(domain_lo[1], domain_hi[1], j, pieces);
            double d1_hi = partition_endpoint(domain_lo[1], domain_hi[1], j + 1, pieces);
            double box_lo, box_hi;
            range_on_box(coefficient_lo, coefficient_hi, order,
                         d0_lo, d0_hi, d1_lo, d1_hi, &box_lo, &box_hi);
            hull_lo = fmin(hull_lo, box_lo);
            hull_hi = fmax(hull_hi, box_hi);
        }
    }
    *result_lo = hull_lo;
    *result_hi = hull_hi;
}

__device__ void cutoff_tm(
    LocalTM *value,
    int order,
    double cutoff,
    int cutoff_enabled,
    const double *domain_lo,
    const double *domain_hi,
    double *cutoff_width
) {
    if (!cutoff_enabled) return;
    int terms = term_count(order);
    double removed_lo[MAX_TERMS];
    double removed_hi[MAX_TERMS];
    bool any = false;
    for (int index = 0; index < terms; ++index) {
        bool remove = fabs(value->point[index]) <= cutoff;
        if (remove && !exact_zero(value->lo[index], value->hi[index])) {
            removed_lo[index] = value->lo[index];
            removed_hi[index] = value->hi[index];
            any = true;
        } else {
            removed_lo[index] = 0.0;
            removed_hi[index] = 0.0;
        }
        if (remove) {
            value->point[index] = 0.0;
            value->lo[index] = 0.0;
            value->hi[index] = 0.0;
        }
    }
    if (any) {
        double lo, hi;
        // Sparse Polynomial.cutoff uses the unsplit natural range.  Split
        // metadata applies only to the dropped truncation polynomial.
        range_polynomial(removed_lo, removed_hi, order, domain_lo, domain_hi,
                         0, &lo, &hi);
        interval_add(value->rem_lo, value->rem_hi, lo, hi,
                     &value->rem_lo, &value->rem_hi);
        *cutoff_width += interval_width(lo, hi);
    }
}

__device__ void multiply_tm_by_inner(
    LocalTM *left,
    const double *right_point,
    const double *right_lo,
    const double *right_hi,
    double right_rem_lo,
    double right_rem_hi,
    int right_split,
    int order,
    double cutoff,
    int cutoff_enabled,
    const double *domain_lo,
    const double *domain_hi,
    double *diagnostics
) {
    int terms = term_count(order);
    int full_terms = term_count(2 * order);
    int drop_terms = full_terms - terms;
    double output_point[MAX_TERMS];
    double output_lo[MAX_TERMS];
    double output_hi[MAX_TERMS];
    double dropped_lo[MAX_DROP_TERMS];
    double dropped_hi[MAX_DROP_TERMS];
    for (int index = 0; index < terms; ++index) {
        output_point[index] = 0.0;
        output_lo[index] = 0.0;
        output_hi[index] = 0.0;
    }
    for (int index = 0; index < drop_terms; ++index) {
        dropped_lo[index] = 0.0;
        dropped_hi[index] = 0.0;
    }

    for (int left_degree = 0; left_degree <= order; ++left_degree) {
        for (int left_e0 = 0; left_e0 <= left_degree; ++left_e0) {
            int left_index = term_index(left_e0, left_degree - left_e0);
            if (exact_zero(left->lo[left_index], left->hi[left_index])) continue;
            for (int right_degree = 0; right_degree <= order; ++right_degree) {
                for (int right_e0 = 0; right_e0 <= right_degree; ++right_e0) {
                    int right_index = term_index(right_e0, right_degree - right_e0);
                    if (exact_zero(right_lo[right_index], right_hi[right_index])) continue;
                    int out_e0 = left_e0 + right_e0;
                    int out_e1 = (left_degree - left_e0) + (right_degree - right_e0);
                    int out_degree = out_e0 + out_e1;
                    double product_lo, product_hi;
                    interval_mul(left->lo[left_index], left->hi[left_index],
                                 right_lo[right_index], right_hi[right_index],
                                 &product_lo, &product_hi);
                    if (out_degree <= order) {
                        int out_index = term_index(out_e0, out_e1);
                        double point_product = left->point[left_index] * right_point[right_index];
                        output_point[out_index] = output_point[out_index] + point_product;
                        interval_add(output_lo[out_index], output_hi[out_index],
                                     product_lo, product_hi,
                                     &output_lo[out_index], &output_hi[out_index]);
                    } else {
                        int drop_index = term_index(out_e0, out_e1) - terms;
                        interval_add(dropped_lo[drop_index], dropped_hi[drop_index],
                                     product_lo, product_hi,
                                     &dropped_lo[drop_index], &dropped_hi[drop_index]);
                    }
                }
            }
        }
    }

    int split = left->split > right_split ? left->split : right_split;
    double left_range_lo, left_range_hi, right_range_lo, right_range_hi;
    range_polynomial(left->lo, left->hi, order, domain_lo, domain_hi,
                     0, &left_range_lo, &left_range_hi);
    range_polynomial(right_lo, right_hi, order, domain_lo, domain_hi,
                     0, &right_range_lo, &right_range_hi);

    double trunc_lo = 0.0;
    double trunc_hi = 0.0;
    if (drop_terms > 0) {
        // Dropped coefficients use the degree-(2*order) tail.  Prefix zeros
        // make the same range evaluator usable without a second reduction law.
        double full_lo[91];
        double full_hi[91];
        for (int index = 0; index < terms; ++index) {
            full_lo[index] = 0.0;
            full_hi[index] = 0.0;
        }
        for (int index = 0; index < drop_terms; ++index) {
            full_lo[terms + index] = dropped_lo[index];
            full_hi[terms + index] = dropped_hi[index];
        }
        range_polynomial(full_lo, full_hi, 2 * order, domain_lo, domain_hi,
                         split, &trunc_lo, &trunc_hi);
    }

    double p_left_j_lo, p_left_j_hi;
    double p_right_i_lo, p_right_i_hi;
    double i_j_lo, i_j_hi;
    interval_mul(left_range_lo, left_range_hi, right_rem_lo, right_rem_hi,
                 &p_left_j_lo, &p_left_j_hi);
    interval_mul(right_range_lo, right_range_hi, left->rem_lo, left->rem_hi,
                 &p_right_i_lo, &p_right_i_hi);
    interval_mul(left->rem_lo, left->rem_hi, right_rem_lo, right_rem_hi,
                 &i_j_lo, &i_j_hi);

    double rem_lo = 0.0;
    double rem_hi = 0.0;
    interval_add(rem_lo, rem_hi, p_left_j_lo, p_left_j_hi, &rem_lo, &rem_hi);
    interval_add(rem_lo, rem_hi, p_right_i_lo, p_right_i_hi, &rem_lo, &rem_hi);
    interval_add(rem_lo, rem_hi, i_j_lo, i_j_hi, &rem_lo, &rem_hi);
    interval_add(rem_lo, rem_hi, trunc_lo, trunc_hi, &rem_lo, &rem_hi);

    for (int index = 0; index < terms; ++index) {
        left->point[index] = output_point[index];
        left->lo[index] = output_lo[index];
        left->hi[index] = output_hi[index];
    }
    left->rem_lo = rem_lo;
    left->rem_hi = rem_hi;
    left->split = split;

    diagnostics[0] += interval_width(trunc_lo, trunc_hi);
    diagnostics[2] += interval_width(p_left_j_lo, p_left_j_hi);
    diagnostics[3] += interval_width(p_right_i_lo, p_right_i_hi);
    diagnostics[4] += interval_width(i_j_lo, i_j_hi);
    diagnostics[6] += 1.0;
    cutoff_tm(left, order, cutoff, cutoff_enabled, domain_lo, domain_hi,
              &diagnostics[1]);
}

__device__ __forceinline__ bool input_coefficient_active(
    double point, double lo, double hi
) {
    return point != 0.0 || !exact_zero(lo, hi);
}

__device__ void set_constant(
    LocalTM *value, int terms, int split,
    double point, double lo, double hi
) {
    zero_tm(value, terms, split);
    value->point[0] = point;
    value->lo[0] = lo;
    value->hi[0] = hi;
}

__device__ void add_constant(
    LocalTM *value, double point, double lo, double hi
) {
    value->point[0] = value->point[0] + point;
    interval_add(value->lo[0], value->hi[0], lo, hi,
                 &value->lo[0], &value->hi[0]);
}

__device__ void add_tm(LocalTM *left, const LocalTM *right, int terms) {
    for (int index = 0; index < terms; ++index) {
        left->point[index] = left->point[index] + right->point[index];
        interval_add(left->lo[index], left->hi[index],
                     right->lo[index], right->hi[index],
                     &left->lo[index], &left->hi[index]);
    }
    interval_add(left->rem_lo, left->rem_hi, right->rem_lo, right->rem_hi,
                 &left->rem_lo, &left->rem_hi);
    left->split = left->split > right->split ? left->split : right->split;
}

__device__ bool build_second_variable_branch(
    LocalTM *branch,
    int first_power,
    const double *outer_point,
    const double *outer_lo,
    const double *outer_hi,
    int outer_split,
    const double *inner_point,
    const double *inner_lo,
    const double *inner_hi,
    const double *inner_rem_lo,
    const double *inner_rem_hi,
    const int *inner_split,
    int order,
    double cutoff,
    int cutoff_enabled,
    const double *domain_lo,
    const double *domain_hi,
    double *diagnostics
) {
    int terms = term_count(order);
    int maximum = -1;
    for (int second_power = 0; second_power + first_power <= order; ++second_power) {
        int index = term_index(first_power, second_power);
        if (input_coefficient_active(outer_point[index], outer_lo[index], outer_hi[index])) {
            maximum = second_power;
        }
    }
    if (maximum < 0) {
        zero_tm(branch, terms, outer_split);
        return false;
    }
    int start = term_index(first_power, maximum);
    set_constant(branch, terms, outer_split,
                 outer_point[start], outer_lo[start], outer_hi[start]);
    const double *right_point = inner_point + terms;
    const double *right_lo = inner_lo + terms;
    const double *right_hi = inner_hi + terms;
    for (int power = maximum - 1; power >= 0; --power) {
        multiply_tm_by_inner(branch, right_point, right_lo, right_hi,
                             inner_rem_lo[1], inner_rem_hi[1], inner_split[1],
                             order, cutoff, cutoff_enabled, domain_lo, domain_hi,
                             diagnostics);
        int index = term_index(first_power, power);
        if (input_coefficient_active(outer_point[index], outer_lo[index], outer_hi[index])) {
            add_constant(branch, outer_point[index], outer_lo[index], outer_hi[index]);
            cutoff_tm(branch, order, cutoff, cutoff_enabled, domain_lo, domain_hi,
                      &diagnostics[1]);
        }
    }
    return true;
}

extern "C" __global__ void resident_normal_compose(
    const double *outer_point,
    const double *outer_lo,
    const double *outer_hi,
    const double *outer_rem_lo,
    const double *outer_rem_hi,
    const double *inner_point,
    const double *inner_lo,
    const double *inner_hi,
    const double *inner_rem_lo,
    const double *inner_rem_hi,
    const double *domain_lo,
    const double *domain_hi,
    const int *outer_split,
    const int *inner_split,
    double *output,
    unsigned long long *receipt,
    int input_stride,
    int batch,
    int outputs,
    int order,
    double cutoff,
    int cutoff_enabled
) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * outputs;
    if (thread >= total) return;
    int lane = thread / outputs;
    int output_index = thread - lane * outputs;
    int terms = term_count(order);
    int stride = 3 * terms + 2 + DIAG_COUNT + 2;
    double *destination = output + thread * stride;
    double diagnostics[DIAG_COUNT];
    for (int index = 0; index < DIAG_COUNT; ++index) diagnostics[index] = 0.0;
    int status = 0;

    const double *lane_outer_point = outer_point + lane * input_stride + output_index * terms;
    const double *lane_outer_lo = outer_lo + lane * input_stride + output_index * terms;
    const double *lane_outer_hi = outer_hi + lane * input_stride + output_index * terms;
    const double *lane_inner_point = inner_point + lane * input_stride;
    const double *lane_inner_lo = inner_lo + lane * input_stride;
    const double *lane_inner_hi = inner_hi + lane * input_stride;
    const double *lane_domain_lo = domain_lo + lane * input_stride;
    const double *lane_domain_hi = domain_hi + lane * input_stride;
    const int *lane_inner_split = inner_split + lane * 2;
    int lane_outer_split = outer_split[thread];

    if (order < 1 || order > MAX_ORDER || outputs < 1 || outputs > 2 ||
        lane_outer_split < 0 || lane_outer_split > MAX_SPLIT ||
        lane_inner_split[0] < 0 || lane_inner_split[0] > MAX_SPLIT ||
        lane_inner_split[1] < 0 || lane_inner_split[1] > MAX_SPLIT ||
        (cutoff_enabled && (!isfinite(cutoff) || cutoff < 0.0))) {
        status |= 1;
    }
    if (!isfinite(lane_domain_lo[0]) || !isfinite(lane_domain_lo[1]) ||
        !isfinite(lane_domain_hi[0]) || !isfinite(lane_domain_hi[1]) ||
        lane_domain_lo[0] > lane_domain_hi[0] || lane_domain_lo[1] > lane_domain_hi[1]) {
        status |= 2;
    }
    if (!isfinite(outer_rem_lo[lane * input_stride + output_index]) ||
        !isfinite(outer_rem_hi[lane * input_stride + output_index]) ||
        outer_rem_lo[lane * input_stride + output_index] >
            outer_rem_hi[lane * input_stride + output_index]) {
        status |= 4;
    }
    for (int index = 0; index < terms; ++index) {
        if (!isfinite(lane_outer_point[index]) || !isfinite(lane_outer_lo[index]) ||
            !isfinite(lane_outer_hi[index]) || lane_outer_lo[index] > lane_outer_hi[index] ||
            lane_outer_point[index] < lane_outer_lo[index] ||
            lane_outer_point[index] > lane_outer_hi[index]) {
            status |= 8;
        }
    }
    for (int variable = 0; variable < 2; ++variable) {
        if (!isfinite(inner_rem_lo[lane * input_stride + variable]) ||
            !isfinite(inner_rem_hi[lane * input_stride + variable]) ||
            inner_rem_lo[lane * input_stride + variable] >
                inner_rem_hi[lane * input_stride + variable]) {
            status |= 16;
        }
        for (int index = 0; index < terms; ++index) {
            int offset = variable * terms + index;
            if (!isfinite(lane_inner_point[offset]) || !isfinite(lane_inner_lo[offset]) ||
                !isfinite(lane_inner_hi[offset]) || lane_inner_lo[offset] > lane_inner_hi[offset] ||
                lane_inner_point[offset] < lane_inner_lo[offset] ||
                lane_inner_point[offset] > lane_inner_hi[offset]) {
                status |= 32;
            }
        }
    }

    LocalTM accumulator;
    zero_tm(&accumulator, terms, lane_outer_split);
    if (status == 0) {
        int maximum = -1;
        for (int first_power = 0; first_power <= order; ++first_power) {
            for (int second_power = 0; first_power + second_power <= order; ++second_power) {
                int index = term_index(first_power, second_power);
                if (input_coefficient_active(lane_outer_point[index], lane_outer_lo[index], lane_outer_hi[index])) {
                    maximum = first_power;
                }
            }
        }
        if (maximum >= 0) {
            build_second_variable_branch(
                &accumulator, maximum, lane_outer_point, lane_outer_lo, lane_outer_hi,
                lane_outer_split, lane_inner_point, lane_inner_lo, lane_inner_hi,
                inner_rem_lo + lane * input_stride,
                inner_rem_hi + lane * input_stride, lane_inner_split,
                order, cutoff, cutoff_enabled, lane_domain_lo, lane_domain_hi,
                diagnostics);
            for (int first_power = maximum - 1; first_power >= 0; --first_power) {
                multiply_tm_by_inner(
                    &accumulator, lane_inner_point, lane_inner_lo, lane_inner_hi,
                    inner_rem_lo[lane * input_stride],
                    inner_rem_hi[lane * input_stride], lane_inner_split[0],
                    order, cutoff, cutoff_enabled, lane_domain_lo, lane_domain_hi,
                    diagnostics);
                LocalTM branch;
                bool present = build_second_variable_branch(
                    &branch, first_power, lane_outer_point, lane_outer_lo, lane_outer_hi,
                    lane_outer_split, lane_inner_point, lane_inner_lo, lane_inner_hi,
                    inner_rem_lo + lane * input_stride,
                    inner_rem_hi + lane * input_stride, lane_inner_split,
                    order, cutoff, cutoff_enabled, lane_domain_lo, lane_domain_hi,
                    diagnostics);
                if (present) {
                    add_tm(&accumulator, &branch, terms);
                    cutoff_tm(&accumulator, order, cutoff, cutoff_enabled,
                              lane_domain_lo, lane_domain_hi, &diagnostics[1]);
                }
            }
        }
        interval_add(accumulator.rem_lo, accumulator.rem_hi,
                     outer_rem_lo[lane * input_stride + output_index],
                     outer_rem_hi[lane * input_stride + output_index],
                     &accumulator.rem_lo, &accumulator.rem_hi);
        diagnostics[7] = interval_width(
            outer_rem_lo[lane * input_stride + output_index],
            outer_rem_hi[lane * input_stride + output_index]);

        // Pay the coefficient error of every retained point coefficient once,
        // after it has been propagated through the complete Horner graph.
        double error_lo[MAX_TERMS];
        double error_hi[MAX_TERMS];
        for (int index = 0; index < terms; ++index) {
            if (accumulator.lo[index] == accumulator.point[index] &&
                accumulator.hi[index] == accumulator.point[index]) {
                error_lo[index] = 0.0;
                error_hi[index] = 0.0;
            } else {
                error_lo[index] = __dsub_rd(accumulator.lo[index], accumulator.point[index]);
                error_hi[index] = __dsub_ru(accumulator.hi[index], accumulator.point[index]);
            }
            destination[terms + index] = error_lo[index];
            destination[2 * terms + index] = error_hi[index];
        }
        double coefficient_error_lo, coefficient_error_hi;
        range_polynomial(error_lo, error_hi, order, lane_domain_lo, lane_domain_hi,
                         accumulator.split, &coefficient_error_lo, &coefficient_error_hi);
        interval_add(accumulator.rem_lo, accumulator.rem_hi,
                     coefficient_error_lo, coefficient_error_hi,
                     &accumulator.rem_lo, &accumulator.rem_hi);
        diagnostics[5] = interval_width(coefficient_error_lo, coefficient_error_hi);

        double point_lo[MAX_TERMS];
        double point_hi[MAX_TERMS];
        for (int index = 0; index < terms; ++index) {
            point_lo[index] = accumulator.point[index];
            point_hi[index] = accumulator.point[index];
        }
        double final_poly_lo, final_poly_hi;
        range_polynomial(point_lo, point_hi, order, lane_domain_lo, lane_domain_hi,
                         0, &final_poly_lo, &final_poly_hi);
        diagnostics[8] = interval_width(final_poly_lo, final_poly_hi);

        if (!isfinite(accumulator.rem_lo) || !isfinite(accumulator.rem_hi) ||
            accumulator.rem_lo > accumulator.rem_hi) {
            status |= 64;
        }
        for (int index = 0; index < terms && status == 0; ++index) {
            if (!isfinite(accumulator.point[index]) || !isfinite(error_lo[index]) ||
                !isfinite(error_hi[index]) || error_lo[index] > error_hi[index]) {
                status |= 64;
            }
        }
    }

    for (int index = 0; index < terms; ++index) {
        destination[index] = accumulator.point[index];
        if (status != 0) {
            destination[terms + index] = 0.0;
            destination[2 * terms + index] = 0.0;
        }
    }
    int base = 3 * terms;
    destination[base] = accumulator.rem_lo;
    destination[base + 1] = accumulator.rem_hi;
    for (int index = 0; index < DIAG_COUNT; ++index) {
        destination[base + 2 + index] = diagnostics[index];
    }
    destination[base + 2 + DIAG_COUNT] = (double)accumulator.split;
    destination[base + 3 + DIAG_COUNT] = (double)status;
    atomicAdd(&receipt[1], 1ULL);
    if (thread == 0) receipt[0] = 0x524553544d424c4bULL;
}

extern "C" __global__ void resident_arithmetic_probe(
    const double *a,
    const double *b,
    double *output,
    int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index * 4] = __dadd_rd(a[index], b[index]);
    output[index * 4 + 1] = __dadd_ru(a[index], b[index]);
    output[index * 4 + 2] = __dmul_rd(a[index], b[index]);
    output[index * 4 + 3] = __dmul_ru(a[index], b[index]);
}
