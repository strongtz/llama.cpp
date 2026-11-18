// v68 qfloat self-test (debug only): compares HVX qfloat sequences with scalar references
#include <HAP_farf.h>
#include <string.h>
#include <math.h>

#include "hvx-base.h"

#if __HVX_ARCH__ < 73

static inline uint16_t hf_bits_p(const __fp16 * x) { uint16_t u; memcpy(&u, x, 2); return u; }

typedef HVX_Vector (*mul_fn)(HVX_Vector, HVX_Vector);

static HVX_Vector mul_qf16(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
}
static HVX_Vector mul_qf16_norm(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(a, b), Q6_V_vzero()));
}
static HVX_Vector mul_qf32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

static void selftest_mul(const char * name, mul_fn f, const __fp16 * a, const __fp16 * b) {
    HVX_Vector va = *(const HVX_Vector *) a, vb = *(const HVX_Vector *) b;
    HVX_Vector r = f(va, vb);
    __fp16 out[64] __attribute__((aligned(128)));
    *(HVX_Vector *) out = r;
    int bad = 0;
    for (int i = 0; i < 64; i++) {
        __fp16 ref = (__fp16) ((float) a[i] * (float) b[i]);
        if (hf_bits_p(&ref) != hf_bits_p(&out[i])) {
            if (bad < 4) FARF(ALWAYS, "  %s: %f * %f = %f (ref %f)", name, (double) a[i], (double) b[i], (double) out[i], (double) ref);
            bad++;
        }
    }
    FARF(ALWAYS, "selftest %s: %d/64 mismatches", name, bad);
}

typedef HVX_Vector (*cvt_fn)(HVX_Vector);

static HVX_Vector cvt_small(HVX_Vector x) { return hvx_v68_hf_from_h_small(x); }
static HVX_Vector cvt_full(HVX_Vector x)  { return hvx_v68_hf_from_h(x); }

static void selftest_cvt(const char * name, cvt_fn f, const int16_t * x) {
    HVX_Vector r = f(*(const HVX_Vector *) x);
    __fp16 out[64] __attribute__((aligned(128)));
    *(HVX_Vector *) out = r;
    int bad = 0;
    for (int i = 0; i < 64; i++) {
        __fp16 ref = (__fp16) (float) x[i];
        if (hf_bits_p(&ref) != hf_bits_p(&out[i])) {
            if (bad < 4) FARF(ALWAYS, "  %s: %d -> %f", name, x[i], (double) out[i]);
            bad++;
        }
    }
    FARF(ALWAYS, "selftest %s: %d/64 mismatches", name, bad);
}

static void selftest_qf32(void) {
    float a[32] __attribute__((aligned(128))), b[32] __attribute__((aligned(128))), out[32] __attribute__((aligned(128)));
    for (int i = 0; i < 32; i++) { a[i] = (float) (i - 16); b[i] = 0.125f; }
    *(HVX_Vector *) out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *) a, *(HVX_Vector *) b));
    int bad = 0;
    for (int i = 0; i < 32; i++) if (out[i] != a[i] * b[i]) { if (bad < 4) FARF(ALWAYS, "  qf32 mul: %f*%f=%f", (double) a[i], (double) b[i], (double) out[i]); bad++; }
    FARF(ALWAYS, "selftest qf32 mul: %d/32 mismatches", bad);
    for (int i = 0; i < 32; i++) { a[i] = 1536.0f + (float) (i - 16); b[i] = 1536.0f; }
    *(HVX_Vector *) out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(*(HVX_Vector *) a, *(HVX_Vector *) b));
    bad = 0;
    for (int i = 0; i < 32; i++) if (out[i] != a[i] - b[i]) { if (bad < 4) FARF(ALWAYS, "  qf32 sub: %f-%f=%f", (double) a[i], (double) b[i], (double) out[i]); bad++; }
    FARF(ALWAYS, "selftest qf32 sub: %d/32 mismatches", bad);
}

void htp_v68_selftest(void) {
    __fp16 a[64] __attribute__((aligned(128))), b[64] __attribute__((aligned(128)));
    int16_t xi[64] __attribute__((aligned(128)));

    for (int i = 0; i < 64; i++) { a[i] = (__fp16) (float) ((i % 16) - 8); b[i] = (__fp16) 0.125f; }
    selftest_mul("qf16 mul (q*0.125)", mul_qf16, a, b);
    selftest_mul("qf16 mul+norm", mul_qf16_norm, a, b);
    selftest_mul("qf32 mul", mul_qf32, a, b);
    for (int i = 0; i < 64; i++) { a[i] = (__fp16) (float) ((i % 32) - 16); b[i] = (__fp16) (0.0137f * (1 + i / 8)); }
    selftest_mul("qf16 mul (q*rand)", mul_qf16, a, b);
    selftest_mul("qf32 mul (q*rand)", mul_qf32, a, b);

    for (int i = 0; i < 64; i++) xi[i] = (int16_t) ((i % 16) - 8);
    selftest_cvt("cvt small", cvt_small, xi);
    selftest_cvt("cvt full", cvt_full, xi);
    for (int i = 0; i < 64; i++) xi[i] = (int16_t) (i * 517 - 16000);
    selftest_cvt("cvt full (wide)", cvt_full, xi);

    selftest_qf32();

    // int32 -> fp32 (reference: round half away from zero == C cast for |x| < 2^24)
    {
        int32_t w[32] __attribute__((aligned(128)));
        float   f[32] __attribute__((aligned(128)));
        int bad = 0;
        uint32_t seed = 12345;
        for (int it = 0; it < 64; it++) {
            for (int i = 0; i < 32; i++) {
                seed = seed * 1664525u + 1013904223u;
                int sh = (seed >> 27);
                w[i] = (int32_t) (seed ^ (seed << 7)) >> sh;
            }
            w[0] = 0; w[1] = -1; w[2] = 0x7FFFFFFF; w[3] = (int32_t) 0x80000000; w[4] = 16777217; w[5] = -16777217;
            *(HVX_Vector *) f = hvx_v68_sf_from_w(*(HVX_Vector *) w);
            for (int i = 0; i < 32; i++) {
                float ref = (float) w[i];
                double err = fabs((double) f[i] - (double) w[i]);
                double ulp = fabs((double) ref) * 1.2e-7;
                if (err > ulp * 0.51 + 1e-30 && f[i] != ref) { if (bad < 4) FARF(ALWAYS, "  sf_from_w: %d -> %f", w[i], (double) f[i]); bad++; }
            }
        }
        FARF(ALWAYS, "selftest sf_from_w: %d/2048 mismatches", bad);

        bad = 0;
        for (int it = 0; it < 64; it++) {
            for (int i = 0; i < 32; i++) {
                seed = seed * 1664525u + 1013904223u;
                f[i] = ((float) (int32_t) seed) / (float) (1u << (seed >> 27));
            }
            f[0] = 0.0f; f[1] = -0.5f; f[2] = 0.99f; f[3] = -1.0f; f[4] = 3e9f; f[5] = -3e9f;
            *(HVX_Vector *) w = hvx_v68_w_from_sf(*(HVX_Vector *) f);
            for (int i = 0; i < 32; i++) {
                int32_t ref = f[i] >= 2147483647.0f ? 0x7FFFFFFF : f[i] <= -2147483648.0f ? (int32_t) 0x80000000 : (int32_t) f[i];
                if (ref != w[i]) { if (bad < 4) FARF(ALWAYS, "  w_from_sf: %f -> %d (ref %d)", (double) f[i], w[i], ref); bad++; }
            }
        }
        FARF(ALWAYS, "selftest w_from_sf: %d/2048 mismatches", bad);
    }
    // fp16 -> int16
    {
        __fp16  h[64] __attribute__((aligned(128)));
        int16_t o[64] __attribute__((aligned(128)));
        int bad = 0;
        for (int it = 0; it < 64; it++) {
            for (int i = 0; i < 64; i++) h[i] = (__fp16) ((float) ((it * 64 + i) * 37 % 70001 - 35000) / (float) (1 + (i % 9)));
            *(HVX_Vector *) o = hvx_v68_h_from_hf(*(HVX_Vector *) h);
            for (int i = 0; i < 64; i++) {
                float v = (float) h[i];
                int16_t ref = v >= 32767.0f ? 32767 : v <= -32768.0f ? -32768 : (int16_t) v;
                if (ref != o[i]) { if (bad < 4) FARF(ALWAYS, "  h_from_hf: %f -> %d (ref %d)", (double) v, o[i], ref); bad++; }
            }
        }
        FARF(ALWAYS, "selftest h_from_hf: %d/4096 mismatches", bad);
    }
}

#else
void htp_v68_selftest(void) {}
#endif
