// Heterogeneous range work packets. Compiled by installed NVRTC, no fast math.
// Requests keep private descriptors, power chains, term chains and ordered sums.
namespace {

enum Header {
    H_MAGIC = 0, H_VERSION = 1, H_BUFFER_EPOCH = 2, H_GLOBAL_STATUS = 3,
    H_RECEIPT_VALIDATE = 4, H_RECEIPT_POWER = 5,
    H_RECEIPT_TERM = 6, H_RECEIPT_SUM = 7,
    H_REQUESTS = 8, H_POWERS = 9, H_CELLS = 10, H_OUTPUTS = 11,
    H_VARIABLES = 12, H_OPERATIONS = 13, H_NUMERIC_LENGTH = 14,
    H_OUTPUT_LENGTH = 15, H_STATUS_OFFSET = 16, H_REQUEST_OFFSET = 17,
    H_POWER_OFFSET = 18, H_CELL_OFFSET = 19, H_OUTPUT_OFFSET = 20,
    H_VARIABLE_OFFSET = 21, H_OPERATION_OFFSET = 22, H_METADATA_LENGTH = 23,
    HEADER_WIDTH = 24
};

enum RequestDescriptor {
    R_TOKEN = 0, R_COEFF_LO = 1, R_COEFF_HI = 2,
    R_DOMAIN_LO = 3, R_DOMAIN_HI = 4,
    R_OUTPUT_START = 5, R_OUTPUT_COUNT = 6,
    R_CELL_START = 7, R_CELL_COUNT = 8,
    R_POWER_START = 9, R_POWER_COUNT = 10,
    R_VARIABLE_START = 11, R_VARIABLE_COUNT = 12,
    R_OPERATION_START = 13, R_OPERATION_COUNT = 14, R_TERMS = 15,
    R_POINT_COEFFICIENTS = 16, R_ENABLED = 17,
    REQUEST_WIDTH = 18
};

enum DescriptorWidths {
    POWER_WIDTH = 3, CELL_WIDTH = 5, OUTPUT_WIDTH = 4, VARIABLE_WIDTH = 3
};

const long long PACKET_MAGIC = 0x5257504b543031LL;  // "RWPKT01"
const long long PACKET_VERSION = 1;
const long long PACKET_INT64_MAX = 9223372036854775807LL;

__device__ bool range_ok(long long offset, long long length, long long limit) {
    return offset >= 0 && length >= 0 && limit >= 0 && offset <= limit && length <= limit - offset;
}

__device__ bool product_ok(long long a, long long b, long long& product) {
    if (a < 0 || b < 0 || (a && b > PACKET_INT64_MAX / a)) return false;
    product = a * b;
    return true;
}

__device__ bool add_ok(long long a, long long b, long long& sum) {
    if (a < 0 || b < 0 || a > PACKET_INT64_MAX - b) return false;
    sum = a + b;
    return true;
}

__device__ unsigned long long* atomic_address(long long* value) {
    return reinterpret_cast<unsigned long long*>(value);
}

__device__ long long atomic_read(long long* value) {
    return static_cast<long long>(atomicAdd(atomic_address(value), 0ULL));
}

__device__ void set_first_status(long long* value, long long code) {
    atomicCAS(atomic_address(value), 0ULL, static_cast<unsigned long long>(code));
}

__device__ bool header_ok(long long* meta, int meta_len, int numeric_len,
                          int output_len, int requests, int work_count,
                          int work_header) {
    if (meta_len < HEADER_WIDTH || numeric_len < 0 || output_len < 0 ||
            requests < 0 || work_count < 0) {
        if (meta_len > H_GLOBAL_STATUS) set_first_status(meta + H_GLOBAL_STATUS, 1);
        return false;
    }
    long long request_items = 0, power_items = 0, cell_items = 0;
    long long output_items = 0, variable_items = 0;
    long long numeric_components = 0, output_components = 0;
    long long expected = HEADER_WIDTH, next = 0, numeric_items = 0, output_values = 0;
    bool products = meta[H_REQUESTS] >= 0 && meta[H_POWERS] >= 0 &&
        meta[H_CELLS] >= 0 && meta[H_OUTPUTS] >= 0 &&
        meta[H_VARIABLES] >= 0 && meta[H_OPERATIONS] >= 0 &&
        product_ok(requests, REQUEST_WIDTH, request_items) &&
        product_ok(meta[H_POWERS], POWER_WIDTH, power_items) &&
        product_ok(meta[H_CELLS], CELL_WIDTH, cell_items) &&
        product_ok(meta[H_OUTPUTS], OUTPUT_WIDTH, output_items) &&
        product_ok(meta[H_VARIABLES], VARIABLE_WIDTH, variable_items) &&
        add_ok(meta[H_CELLS], meta[H_VARIABLES], numeric_components) &&
        product_ok(numeric_components, 2, numeric_items) &&
        add_ok(meta[H_OUTPUTS], meta[H_CELLS], output_components) &&
        add_ok(output_components, meta[H_POWERS], output_components) &&
        product_ok(output_components, 2, output_values);
    bool layout = products && meta[H_STATUS_OFFSET] == expected &&
        add_ok(expected, requests, next);
    expected = next;
    layout = layout && meta[H_REQUEST_OFFSET] == expected && add_ok(expected, request_items, next);
    expected = next;
    layout = layout && meta[H_POWER_OFFSET] == expected && add_ok(expected, power_items, next);
    expected = next;
    layout = layout && meta[H_CELL_OFFSET] == expected && add_ok(expected, cell_items, next);
    expected = next;
    layout = layout && meta[H_OUTPUT_OFFSET] == expected && add_ok(expected, output_items, next);
    expected = next;
    layout = layout && meta[H_VARIABLE_OFFSET] == expected && add_ok(expected, variable_items, next);
    expected = next;
    layout = layout && meta[H_OPERATION_OFFSET] == expected &&
        add_ok(expected, meta[H_OPERATIONS], next) && next == meta_len;
    bool ok = layout &&
        meta[H_MAGIC] == PACKET_MAGIC && meta[H_VERSION] == PACKET_VERSION &&
        meta[H_REQUESTS] == requests && work_header >= H_REQUESTS &&
        work_header <= H_OUTPUTS && meta[work_header] == work_count &&
        meta[H_METADATA_LENGTH] == meta_len &&
        meta[H_NUMERIC_LENGTH] == numeric_len && meta[H_OUTPUT_LENGTH] == output_len &&
        numeric_items == numeric_len && output_values == output_len &&
        range_ok(meta[H_STATUS_OFFSET], requests, meta_len) &&
        range_ok(meta[H_REQUEST_OFFSET], request_items, meta_len) &&
        range_ok(meta[H_POWER_OFFSET], power_items, meta_len) &&
        range_ok(meta[H_CELL_OFFSET], cell_items, meta_len) &&
        range_ok(meta[H_OUTPUT_OFFSET], output_items, meta_len) &&
        range_ok(meta[H_VARIABLE_OFFSET], variable_items, meta_len) &&
        range_ok(meta[H_OPERATION_OFFSET], meta[H_OPERATIONS], meta_len);
    if (!ok) set_first_status(meta + H_GLOBAL_STATUS, 1);
    return ok && atomic_read(meta + H_GLOBAL_STATUS) == 0;
}

__device__ bool subrange_ok(long long offset, long long length,
                            long long owner_offset, long long owner_length) {
    if (offset < owner_offset || length < 0 || owner_offset < 0 || owner_length < 0) return false;
    long long local = offset - owner_offset;
    return local <= owner_length && length <= owner_length - local;
}

__device__ long long* request_descriptor(long long* meta, int request) {
    return meta + meta[H_REQUEST_OFFSET] + static_cast<long long>(request) * REQUEST_WIDTH;
}

__device__ long long* request_status(long long* meta, int request) {
    return meta + meta[H_STATUS_OFFSET] + request;
}

__device__ int partition_owner(long long* meta, long long index,
                               int count_field, int requests) {
    long long cursor = 0, next = 0;
    for (int owner = 0; owner < requests; ++owner) {
        long long count = request_descriptor(meta, owner)[count_field];
        if (!add_ok(cursor, count, next)) return -1;
        if (index >= cursor && index < next) return owner;
        cursor = next;
    }
    return -1;
}

__device__ void positive_power(double x, int n, double& lo, double& hi) {
    lo = 1.; hi = 1.;
    for (int i = 0; i < n; ++i) {
        lo = __dmul_rd(lo, x);
        hi = __dmul_ru(hi, x);
    }
}

__device__ void endpoint_power(double x, int n, double& lo, double& hi) {
    positive_power(fabs(x), n, lo, hi);
    if (x < 0. && (n & 1)) {
        double temp = lo; lo = -hi; hi = -temp;
    }
}

__device__ void interval_product(double a, double z, double l, double h,
                                 double& out_l, double& out_h) {
    out_l = fmin(fmin(__dmul_rd(a,l), __dmul_rd(a,h)),
                 fmin(__dmul_rd(z,l), __dmul_rd(z,h)));
    out_h = fmax(fmax(__dmul_ru(a,l), __dmul_ru(a,h)),
                 fmax(__dmul_ru(z,l), __dmul_ru(z,h)));
}

}  // namespace

extern "C" __global__ void packet_validate(
    long long* meta, const double* numeric, double* output,
    int meta_len, int numeric_len, int output_len, int requests) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b == 0 && meta_len > H_RECEIPT_VALIDATE)
        atomicAdd(atomic_address(meta + H_RECEIPT_VALIDATE), 1ULL);
    if (!header_ok(meta, meta_len, numeric_len, output_len, requests, requests,
                   H_REQUESTS) || b >= requests) return;

    // Request section lengths must exactly cover the packet.  Offset ownership
    // is checked per request below so one damaged descriptor can be isolated.
    if (b == 0) {
        long long cells = 0, outputs = 0, powers = 0, variables = 0, operations = 0;
        bool partition = true;
        for (int owner = 0; owner < requests && partition; ++owner) {
            long long* q = request_descriptor(meta, owner);
            partition = q[R_OUTPUT_COUNT] >= 0 && q[R_CELL_COUNT] >= 0 &&
                q[R_POWER_COUNT] >= 0 &&
                q[R_VARIABLE_COUNT] >= 0 && q[R_OPERATION_COUNT] >= 0 &&
                add_ok(cells, q[R_CELL_COUNT], cells) &&
                add_ok(outputs, q[R_OUTPUT_COUNT], outputs) &&
                add_ok(powers, q[R_POWER_COUNT], powers) &&
                add_ok(variables, q[R_VARIABLE_COUNT], variables) &&
                add_ok(operations, q[R_OPERATION_COUNT], operations);
        }
        partition = partition && cells == meta[H_CELLS] &&
            outputs == meta[H_OUTPUTS] && powers == meta[H_POWERS] &&
            variables == meta[H_VARIABLES] && operations == meta[H_OPERATIONS];
        if (!partition) set_first_status(meta + H_GLOBAL_STATUS, 5);
    }
    long long* status = request_status(meta, b);
    long long* r = request_descriptor(meta, b);
    long long prefix_cells = 0, prefix_outputs = 0, prefix_powers = 0;
    long long prefix_variables = 0, prefix_operations = 0;
    bool prefix_ok = true;
    for (int owner = 0; owner < b && prefix_ok; ++owner) {
        long long* q = request_descriptor(meta, owner);
        prefix_ok = add_ok(prefix_cells, q[R_CELL_COUNT], prefix_cells) &&
            add_ok(prefix_outputs, q[R_OUTPUT_COUNT], prefix_outputs) &&
            add_ok(prefix_powers, q[R_POWER_COUNT], prefix_powers) &&
            add_ok(prefix_variables, q[R_VARIABLE_COUNT], prefix_variables) &&
            add_ok(prefix_operations, q[R_OPERATION_COUNT], prefix_operations);
    }
    long long cells, expected_coeff_hi = 0;
    long long twice_cells = 0, expected_domain_lo = 0;
    long long domain_hi_base = 0, expected_domain_hi = 0;
    bool ok = r[R_TOKEN] >= 0 && r[R_OUTPUT_COUNT] > 0 && r[R_TERMS] >= 0 &&
        r[R_VARIABLE_COUNT] >= 0 && (r[R_POINT_COEFFICIENTS] == 0 || r[R_POINT_COEFFICIENTS] == 1) &&
        (r[R_ENABLED] == 0 || r[R_ENABLED] == 1) &&
        product_ok(r[R_OUTPUT_COUNT], r[R_TERMS], cells) &&
        cells == r[R_CELL_COUNT] &&
        prefix_ok && add_ok(meta[H_CELLS], prefix_cells, expected_coeff_hi) &&
        product_ok(meta[H_CELLS], 2, twice_cells) &&
        add_ok(twice_cells, prefix_variables, expected_domain_lo) &&
        add_ok(twice_cells, meta[H_VARIABLES], domain_hi_base) &&
        add_ok(domain_hi_base, prefix_variables, expected_domain_hi) &&
        r[R_COEFF_LO] == prefix_cells && r[R_COEFF_HI] == expected_coeff_hi &&
        r[R_DOMAIN_LO] == expected_domain_lo && r[R_DOMAIN_HI] == expected_domain_hi &&
        r[R_OUTPUT_START] == prefix_outputs && r[R_CELL_START] == prefix_cells &&
        r[R_POWER_START] == prefix_powers && r[R_VARIABLE_START] == prefix_variables &&
        r[R_OPERATION_START] == prefix_operations &&
        range_ok(r[R_COEFF_LO], cells, numeric_len) && range_ok(r[R_COEFF_HI], cells, numeric_len) &&
        range_ok(r[R_DOMAIN_LO], r[R_VARIABLE_COUNT], numeric_len) &&
        range_ok(r[R_DOMAIN_HI], r[R_VARIABLE_COUNT], numeric_len) &&
        range_ok(r[R_OUTPUT_START], r[R_OUTPUT_COUNT], meta[H_OUTPUTS]) &&
        range_ok(r[R_CELL_START], cells, meta[H_CELLS]) &&
        range_ok(r[R_POWER_START], r[R_POWER_COUNT], meta[H_POWERS]) &&
        range_ok(r[R_VARIABLE_START], r[R_VARIABLE_COUNT], meta[H_VARIABLES]) &&
        range_ok(r[R_OPERATION_START], r[R_OPERATION_COUNT], meta[H_OPERATIONS]);
    if (!ok) { set_first_status(status, 4); return; }
    if (!r[R_ENABLED]) { set_first_status(status, 2); return; }

    for (long long v = 0; v < r[R_VARIABLE_COUNT]; ++v) {
        long long* d = meta + meta[H_VARIABLE_OFFSET] + (r[R_VARIABLE_START] + v) * VARIABLE_WIDTH;
        if (d[0] != b || d[1] != v || (d[2] != 0 && d[2] != 1)) {
            set_first_status(status, 4); return;
        }
        double lo = numeric[r[R_DOMAIN_LO] + v], hi = numeric[r[R_DOMAIN_HI] + v];
        if (!isfinite(lo) || !isfinite(hi) || lo > hi || (d[2] && (lo < -1. || hi > 1.))) {
            set_first_status(status, 1); return;
        }
    }
    for (long long i = 0; i < cells; ++i) {
        double lo = numeric[r[R_COEFF_LO] + i], hi = numeric[r[R_COEFF_HI] + i];
        if (!isfinite(lo) || !isfinite(hi) || lo > hi || (r[R_POINT_COEFFICIENTS] && lo != hi)) {
            set_first_status(status, 1); return;
        }
    }
    for (long long p = 0; p < r[R_POWER_COUNT]; ++p) {
        long long* d = meta + meta[H_POWER_OFFSET] + (r[R_POWER_START] + p) * POWER_WIDTH;
        if (d[0] != b || d[1] < 0 || d[1] >= r[R_VARIABLE_COUNT] || d[2] < 0 || d[2] > 64) {
            set_first_status(status, 4); return;
        }
    }
    for (long long o = 0; o < r[R_OUTPUT_COUNT]; ++o) {
        long long* d = meta + meta[H_OUTPUT_OFFSET] + (r[R_OUTPUT_START] + o) * OUTPUT_WIDTH;
        if (d[0] != b || d[1] != o || d[2] != r[R_CELL_START] + o * r[R_TERMS] || d[3] != r[R_TERMS]) {
            set_first_status(status, 4); return;
        }
    }
    for (long long c = 0; c < cells; ++c) {
        long long* d = meta + meta[H_CELL_OFFSET] + (r[R_CELL_START] + c) * CELL_WIDTH;
        long long expected_o = r[R_TERMS] ? c / r[R_TERMS] : 0;
        long long expected_t = r[R_TERMS] ? c % r[R_TERMS] : 0;
        if (d[0] != b || d[1] != expected_o || d[2] != expected_t ||
                !range_ok(d[3], d[4], meta[H_OPERATIONS]) ||
                !subrange_ok(d[3], d[4], r[R_OPERATION_START], r[R_OPERATION_COUNT])) {
            set_first_status(status, 4); return;
        }
        for (long long j = 0; j < d[4]; ++j) {
            long long op = meta[meta[H_OPERATION_OFFSET] + d[3] + j];
            if (op >= meta[H_POWERS] || op < -4 || op == -1 ||
                    (op >= 0 && !subrange_ok(op, 1, r[R_POWER_START], r[R_POWER_COUNT]))) {
                set_first_status(status, 4); return;
            }
            if (op >= 0) {
                long long* power = meta + meta[H_POWER_OFFSET] + op * POWER_WIDTH;
                if (power[0] != b) { set_first_status(status, 4); return; }
            }
        }
    }
}

extern "C" __global__ void packet_powers(
    long long* meta, const double* numeric, double* output,
    int meta_len, int numeric_len, int output_len, int requests, int powers) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0 && meta_len > H_RECEIPT_POWER)
        atomicAdd(atomic_address(meta + H_RECEIPT_POWER), 1ULL);
    if (!header_ok(meta, meta_len, numeric_len, output_len, requests, powers,
                   H_POWERS) || index >= powers) return;
    long long* d = meta + meta[H_POWER_OFFSET] + static_cast<long long>(index) * POWER_WIDTH;
    int b = -1;
    if (d[0] >= 0 && d[0] < requests) {
        int claimed = static_cast<int>(d[0]);
        long long* claimed_request = request_descriptor(meta, claimed);
        if (subrange_ok(index, 1, claimed_request[R_POWER_START],
                        claimed_request[R_POWER_COUNT])) b = claimed;
    }
    if (b < 0) b = partition_owner(meta, index, R_POWER_COUNT, requests);
    if (b < 0) { set_first_status(meta + H_GLOBAL_STATUS, 2); return; }
    long long* status = request_status(meta, b);
    long long power_lo_base = 2 * meta[H_OUTPUTS] + 2 * meta[H_CELLS];
    long long power_hi_base = power_lo_base + meta[H_POWERS];
    if (atomic_read(status)) {
        output[power_lo_base + index] = nan(""); output[power_hi_base + index] = nan(""); return;
    }
    long long* r = request_descriptor(meta, b);
    if (d[0] != b || !subrange_ok(index, 1, r[R_POWER_START], r[R_POWER_COUNT])) {
        set_first_status(status, 4); return;
    }
    int variable = static_cast<int>(d[1]), exponent = static_cast<int>(d[2]);
    if (variable < 0 || variable >= r[R_VARIABLE_COUNT] || exponent < 0 || exponent > 64) {
        set_first_status(status, 4); return;
    }
    double a = numeric[r[R_DOMAIN_LO] + variable];
    double z = numeric[r[R_DOMAIN_HI] + variable];
    double lo, hi;
    if (exponent == 0) { lo = 1.; hi = 1.; }
    else if (exponent == 1) { lo = a; hi = z; }
    else {
        double al, ah, zl, zh;
        endpoint_power(a, exponent, al, ah); endpoint_power(z, exponent, zl, zh);
        lo = ((exponent % 2 == 0) && a <= 0. && z >= 0.) ? 0. : fmin(al, zl);
        hi = fmax(ah, zh);
    }
    output[power_lo_base + index] = lo; output[power_hi_base + index] = hi;
    if (!isfinite(lo) || !isfinite(hi) || lo > hi) set_first_status(status, 3);
}

extern "C" __global__ void packet_terms(
    long long* meta, const double* numeric, double* output,
    int meta_len, int numeric_len, int output_len, int requests, int cells) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0 && meta_len > H_RECEIPT_TERM)
        atomicAdd(atomic_address(meta + H_RECEIPT_TERM), 1ULL);
    if (!header_ok(meta, meta_len, numeric_len, output_len, requests, cells,
                   H_CELLS) || index >= cells) return;
    long long* d = meta + meta[H_CELL_OFFSET] + static_cast<long long>(index) * CELL_WIDTH;
    int b = -1;
    if (d[0] >= 0 && d[0] < requests) {
        int claimed = static_cast<int>(d[0]);
        long long* claimed_request = request_descriptor(meta, claimed);
        if (subrange_ok(index, 1, claimed_request[R_CELL_START],
                        claimed_request[R_CELL_COUNT])) b = claimed;
    }
    if (b < 0) b = partition_owner(meta, index, R_CELL_COUNT, requests);
    if (b < 0) { set_first_status(meta + H_GLOBAL_STATUS, 3); return; }
    long long* status = request_status(meta, b);
    long long term_lo_base = 2 * meta[H_OUTPUTS];
    long long term_hi_base = term_lo_base + meta[H_CELLS];
    if (atomic_read(status)) {
        output[term_lo_base + index] = nan(""); output[term_hi_base + index] = nan(""); return;
    }
    long long* r = request_descriptor(meta, b);
    long long local_output = d[1], local_term = d[2];
    if (d[0] != b || local_output < 0 || local_output >= r[R_OUTPUT_COUNT] ||
            local_term < 0 || local_term >= r[R_TERMS] ||
            !subrange_ok(index, 1, r[R_CELL_START], r[R_CELL_COUNT]) ||
            !range_ok(d[3], d[4], meta[H_OPERATIONS]) ||
            !subrange_ok(d[3], d[4], r[R_OPERATION_START], r[R_OPERATION_COUNT])) {
        set_first_status(status, 4); return;
    }
    long long coefficient = local_output * r[R_TERMS] + local_term;
    double a = numeric[r[R_COEFF_LO] + coefficient];
    double z = numeric[r[R_COEFF_HI] + coefficient];
    long long power_lo_base = 2 * meta[H_OUTPUTS] + 2 * meta[H_CELLS];
    long long power_hi_base = power_lo_base + meta[H_POWERS];
    for (long long j = 0; j < d[4]; ++j) {
        long long op = meta[meta[H_OPERATION_OFFSET] + d[3] + j];
        double lo, hi;
        if (op >= 0 && op < meta[H_POWERS] &&
                subrange_ok(op, 1, r[R_POWER_START], r[R_POWER_COUNT])) {
            lo = output[power_lo_base + op]; hi = output[power_hi_base + op];
        } else if (op == -2) { lo = -1.; hi = 1.; }
        else if (op == -3) { lo = 0.; hi = 1.; }
        else if (op == -4) { lo = 1.; hi = 1.; }
        else { set_first_status(status, 4); return; }
        double next_lo, next_hi;
        interval_product(a, z, lo, hi, next_lo, next_hi);
        a = next_lo; z = next_hi;
        if (!isfinite(a) || !isfinite(z) || a > z) { set_first_status(status, 3); break; }
    }
    output[term_lo_base + index] = a; output[term_hi_base + index] = z;
}

extern "C" __global__ void packet_sum(
    long long* meta, const double* numeric, double* output,
    int meta_len, int numeric_len, int output_len, int requests, int outputs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0 && meta_len > H_RECEIPT_SUM)
        atomicAdd(atomic_address(meta + H_RECEIPT_SUM), 1ULL);
    if (!header_ok(meta, meta_len, numeric_len, output_len, requests, outputs,
                   H_OUTPUTS) || index >= outputs) return;
    long long* d = meta + meta[H_OUTPUT_OFFSET] + static_cast<long long>(index) * OUTPUT_WIDTH;
    int b = -1;
    if (d[0] >= 0 && d[0] < requests) {
        int claimed = static_cast<int>(d[0]);
        long long* claimed_request = request_descriptor(meta, claimed);
        if (subrange_ok(index, 1, claimed_request[R_OUTPUT_START],
                        claimed_request[R_OUTPUT_COUNT])) b = claimed;
    }
    if (b < 0) b = partition_owner(meta, index, R_OUTPUT_COUNT, requests);
    if (b < 0) { set_first_status(meta + H_GLOBAL_STATUS, 4); return; }
    long long* status = request_status(meta, b);
    if (atomic_read(status)) {
        output[index] = nan(""); output[meta[H_OUTPUTS] + index] = nan(""); return;
    }
    long long term_lo_base = 2 * meta[H_OUTPUTS];
    long long term_hi_base = term_lo_base + meta[H_CELLS];
    long long* r = request_descriptor(meta, b);
    if (d[0] != b || !subrange_ok(index, 1, r[R_OUTPUT_START], r[R_OUTPUT_COUNT]) ||
            !range_ok(d[2], d[3], meta[H_CELLS]) ||
            !subrange_ok(d[2], d[3], r[R_CELL_START], r[R_CELL_COUNT])) {
        set_first_status(status, 4); return;
    }
    double lo = 0., hi = 0.;
    for (long long term = 0; term < d[3]; ++term) {
        lo = __dadd_rd(lo, output[term_lo_base + d[2] + term]);
        hi = __dadd_ru(hi, output[term_hi_base + d[2] + term]);
        if (!isfinite(lo) || !isfinite(hi) || lo > hi) { set_first_status(status, 3); break; }
    }
    output[index] = lo; output[meta[H_OUTPUTS] + index] = hi;
}

extern "C" __global__ void packet_arithmetic_probe(
    const double* a, const double* b, double* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[4*index] = __dadd_rd(a[index], b[index]);
    output[4*index+1] = __dadd_ru(a[index], b[index]);
    output[4*index+2] = __dmul_rd(a[index], b[index]);
    output[4*index+3] = __dmul_ru(a[index], b[index]);
}
