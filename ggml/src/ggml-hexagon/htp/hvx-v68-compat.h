#ifndef HVX_V68_COMPAT_H
#define HVX_V68_COMPAT_H

// Emulation of HVX conversion instructions that were introduced in v73
// (vconv_sf_w, vconv_w_sf, vconv_hf_h, vconv_h_hf) for v68/v69 targets.
//
// Note: v68 qfloat arithmetic is not exact (results can be off by ~1 ulp of the
// larger operand), so the usual "add magic constant then subtract" tricks do not
// work. Everything below uses integer bit manipulation only.

#include <hexagon_types.h>
#include <hexagon_protos.h>

#if defined(__HVX_ARCH__) && __HVX_ARCH__ < 73

// int32 -> fp32, round half away from zero
static inline HVX_Vector hvx_v68_sf_from_w(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    const HVX_Vector c8   = Q6_V_vsplat_R(8);

    HVX_Vector a   = Q6_Vw_vabs_Vw(x);                          // INT_MIN stays 0x80000000 (ok as unsigned)
    HVX_Vector lz  = Q6_Vuw_vcl0_Vuw(a);
    HVX_Vector sl  = Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(lz, c8), zero);
    HVX_Vector sr  = Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(c8, lz), zero);
    // rounding bias for right shifts: 1 << (sr - 1) when sr > 0
    HVX_Vector rb  = Q6_Vw_vasl_VwVw(one, Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(sr, one), zero));
    rb             = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(sr, zero), rb, zero);
    HVX_Vector m   = Q6_Vw_vlsr_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwVw(a, sl), rb), sr);
    HVX_Vector ex  = Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(157), lz), 23);
    HVX_Vector r   = Q6_Vw_vadd_VwVw(ex, m);                    // mantissa carry bumps the exponent
    r              = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(a, zero), zero, r);
    return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_V_vsplat_R(0x80000000)));
}

// fp32 -> int32, truncate toward zero, saturating
static inline HVX_Vector hvx_v68_w_from_sf(HVX_Vector x) {
    const HVX_Vector c0     = Q6_V_vzero();
    const HVX_Vector c31    = Q6_V_vsplat_R(31);
    const HVX_Vector c150   = Q6_V_vsplat_R(150);
    const HVX_Vector c157   = Q6_V_vsplat_R(157);
    const HVX_Vector c149   = Q6_V_vsplat_R(149);
    const HVX_Vector mmask  = Q6_V_vsplat_R(0x007FFFFF);
    const HVX_Vector mimpl  = Q6_V_vsplat_R(0x00800000);
    const HVX_Vector emask  = Q6_V_vsplat_R(0xFF);
    const HVX_Vector imax   = Q6_V_vsplat_R(0x7FFFFFFF);
    const HVX_Vector imin   = Q6_V_vsplat_R(0x80000000);

    HVX_Vector e  = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x, 23), emask);
    HVX_Vector m  = Q6_V_vor_VV(Q6_V_vand_VV(x, mmask), mimpl);

    HVX_Vector l  = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(e, c150), c0), c31);
    HVX_Vector r  = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(c150, e), c0), c31);

    HVX_Vector vl = Q6_Vw_vasl_VwVw(m, l);
    HVX_Vector vr = Q6_Vw_vlsr_VwVw(m, r);

    HVX_VectorPred q_left = Q6_Q_vcmp_gt_VwVw(e, c149);
    HVX_Vector     v      = Q6_V_vmux_QVV(q_left, vl, vr);

    HVX_VectorPred q_neg  = Q6_Q_vcmp_gt_VwVw(c0, x);
    v                     = Q6_V_vmux_QVV(q_neg, Q6_Vw_vsub_VwVw(c0, v), v);

    HVX_VectorPred q_big  = Q6_Q_vcmp_gt_VwVw(e, c157);
    HVX_Vector     sat    = Q6_V_vmux_QVV(q_neg, imin, imax);
    return Q6_V_vmux_QVV(q_big, sat, v);
}

// int16 -> fp16, round half away from zero
static inline HVX_Vector hvx_v68_hf_from_h(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_Vh_vsplat_R(1);
    const HVX_Vector c5   = Q6_Vh_vsplat_R(5);

    HVX_Vector a   = Q6_Vh_vabs_Vh(x);                          // -32768 stays 0x8000 (ok as unsigned)
    HVX_Vector lz  = Q6_Vuh_vcl0_Vuh(a);
    HVX_Vector sl  = Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(lz, c5), zero);
    HVX_Vector sr  = Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(c5, lz), zero);
    HVX_Vector rb  = Q6_Vh_vasl_VhVh(one, Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(sr, one), zero));
    rb             = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(sr, zero), rb, zero);
    HVX_Vector m   = Q6_Vh_vlsr_VhVh(Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhVh(a, sl), rb), sr);
    HVX_Vector ex  = Q6_Vh_vasl_VhR(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(29), lz), 10);
    HVX_Vector r   = Q6_Vh_vadd_VhVh(ex, m);
    r              = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(a, zero), zero, r);
    return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_Vh_vsplat_R(0x8000)));
}

// int16 -> fp16 for |x| < 2048 (exact, no rounding needed)
static inline HVX_Vector hvx_v68_hf_from_h_small(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector a   = Q6_Vh_vabs_Vh(x);
    HVX_Vector lz  = Q6_Vuh_vcl0_Vuh(a);
    HVX_Vector m   = Q6_Vh_vasl_VhVh(a, Q6_Vh_vsub_VhVh(lz, Q6_Vh_vsplat_R(5)));
    HVX_Vector ex  = Q6_Vh_vasl_VhR(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(29), lz), 10);
    HVX_Vector r   = Q6_Vh_vadd_VhVh(ex, m);
    r              = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(a, zero), zero, r);
    return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_Vh_vsplat_R(0x8000)));
}

// fp16 -> int16, truncate toward zero, saturating
static inline HVX_Vector hvx_v68_h_from_hf(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector c25  = Q6_Vh_vsplat_R(25);
    const HVX_Vector c15  = Q6_Vh_vsplat_R(15);

    HVX_Vector e  = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 10), Q6_Vh_vsplat_R(0x1F));
    HVX_Vector m  = Q6_V_vor_VV(Q6_V_vand_VV(x, Q6_Vh_vsplat_R(0x3FF)), Q6_Vh_vsplat_R(0x400));
    HVX_Vector sl = Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(e, c25), zero);
    HVX_Vector sr = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(c25, e), zero), c15);
    HVX_Vector v  = Q6_Vh_vlsr_VhVh(Q6_Vh_vasl_VhVh(m, sl), sr);

    HVX_VectorPred q_neg = Q6_Q_vcmp_gt_VhVh(zero, x);
    v = Q6_V_vmux_QVV(q_neg, Q6_Vh_vsub_VhVh(zero, v), v);

    HVX_VectorPred q_big = Q6_Q_vcmp_gt_VhVh(e, Q6_Vh_vsplat_R(29));   // |x| >= 32768
    HVX_Vector     sat   = Q6_V_vmux_QVV(q_neg, Q6_Vh_vsplat_R(0x8000), Q6_Vh_vsplat_R(0x7FFF));
    return Q6_V_vmux_QVV(q_big, sat, v);
}

#define Q6_Vsf_equals_Vw(v) hvx_v68_sf_from_w(v)
#define Q6_Vw_equals_Vsf(v) hvx_v68_w_from_sf(v)
#define Q6_Vhf_equals_Vh(v) hvx_v68_hf_from_h(v)
#define Q6_Vh_equals_Vhf(v) hvx_v68_h_from_hf(v)

#endif  // __HVX_ARCH__ < 73

#endif  // HVX_V68_COMPAT_H
