// Local range operator only. Compiled by the installed NVRTC; no fast math.
// Each request owns its status, all numeric buffers and its ordered accumulation.
extern "C" __global__ void validate_rows(
    const double* cl, const double* ch, const double* dl, const double* dh,
    const int* states, const unsigned char* mask, int* status,
    unsigned long long* receipt, int B, int O, int T, int V, int point_coeff) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b == 0) atomicAdd(receipt, 1ULL);
    if (b >= B) return;
    if (!mask[b]) { status[b] = 2; return; }
    int code = 0;
    for (int v = 0; v < V; ++v) {
        double l = dl[b*V+v], h = dh[b*V+v];
        if (!isfinite(l) || !isfinite(h) || l > h || (states[v] && (l < -1. || h > 1.))) code = 1;
    }
    for (int i = 0; i < O*T; ++i) {
        double l = cl[b*O*T+i], h = ch[b*O*T+i];
        if (!isfinite(l) || !isfinite(h) || l > h || (point_coeff && l != h)) code = 1;
    }
    status[b] = code;
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
    if (x < 0. && (n & 1)) { double temp = lo; lo = -hi; hi = -temp; }
}

extern "C" __global__ void range_powers(
    const double* dl, const double* dh, const int* keys, double* pl, double* ph,
    int* status, unsigned long long* receipt, int B, int P, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) atomicAdd(receipt+1, 1ULL);
    if (i >= B*P) return;
    int b = i/P, key = i%P;
    if (atomicAdd(status+b,0)) { pl[i] = nan(""); ph[i] = nan(""); return; }
    int v = keys[2*key], p = keys[2*key+1];
    double a = dl[b*V+v], z = dh[b*V+v], l, h;
    if (p == 0) { l = 1.; h = 1.; }
    else if (p == 1) { l = a; h = z; }
    else {
        double al, ah, zl, zh;
        endpoint_power(a,p,al,ah); endpoint_power(z,p,zl,zh);
        l = ((p%2 == 0) && a <= 0. && z >= 0.) ? 0. : fmin(al,zl);
        h = fmax(ah,zh);
    }
    pl[i] = l; ph[i] = h;
    if (!isfinite(l) || !isfinite(h) || l > h) atomicExch(status+b,3);
}

__device__ void interval_product(double a, double z, double l, double h, double& out_l, double& out_h) {
    out_l = fmin(fmin(__dmul_rd(a,l),__dmul_rd(a,h)),fmin(__dmul_rd(z,l),__dmul_rd(z,h)));
    out_h = fmax(fmax(__dmul_ru(a,l),__dmul_ru(a,h)),fmax(__dmul_ru(z,l),__dmul_ru(z,h)));
}

extern "C" __global__ void range_terms(
    const double* cl, const double* ch, const double* pl, const double* ph,
    const int* ops, double* tl, double* th, int* status,
    unsigned long long* receipt, int B, int O, int T, int P, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) atomicAdd(receipt+2, 1ULL);
    if (i >= B*O*T) return;
    int b = i/(O*T), term = i%T;
    if (atomicAdd(status+b,0)) { tl[i] = nan(""); th[i] = nan(""); return; }
    double a = cl[i], z = ch[i];
    for (int s = 0; s < S; ++s) {
        int op = ops[s*T+term];
        if (op == -1) continue;
        double l, h;
        if (op >= 0) { l = pl[b*P+op]; h = ph[b*P+op]; }
        else { l = (op == -2 ? -1. : op == -3 ? 0. : 1.); h = 1.; }
        double low, high;
        interval_product(a,z,l,h,low,high);
        a = low; z = high;
        if (!isfinite(a) || !isfinite(z) || a > z) { atomicExch(status+b,3); break; }
    }
    tl[i] = a; th[i] = z;
}

extern "C" __global__ void range_sum(
    const double* tl, const double* th, double* lo, double* hi,
    int* status, unsigned long long* receipt, int B, int O, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) atomicAdd(receipt+3, 1ULL);
    if (i >= B*O) return;
    int b = i/O;
    if (atomicAdd(status+b,0)) { lo[i] = nan(""); hi[i] = nan(""); return; }
    double l = 0., h = 0.;
    for (int t = 0; t < T; ++t) {
        l = __dadd_rd(l,tl[i*T+t]);
        h = __dadd_ru(h,th[i*T+t]);
        if (!isfinite(l) || !isfinite(h) || l > h) { atomicExch(status+b,3); break; }
    }
    lo[i] = l; hi[i] = h;
}

extern "C" __global__ void arithmetic_probe(const double* a, const double* b, double* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[4*i] = __dadd_rd(a[i],b[i]); out[4*i+1] = __dadd_ru(a[i],b[i]);
    out[4*i+2] = __dmul_rd(a[i],b[i]); out[4*i+3] = __dmul_ru(a[i],b[i]);
}
