// MLA single-query flash attention (decode) for heads sharing one KV head.
// DK = 576 (512 latent + 64 rope), DV = 512, V is (a view of) the first DV elements of K rows.
// Workgroup = NH/HG subgroups of 64 lanes, subgroup y handles heads [y*HG, y*HG+HG) for the
// same KV chunk (K/V rows are shared through the cache). Each lane handles RPL rows per block.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SG64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SG64
#endif

#ifndef NH
#define NH 16
#endif
#ifndef DK
#define DK 576
#endif
#ifndef DV
#define DV 512
#endif
#ifndef HG
#define HG 4      // heads per subgroup
#endif
#ifndef RPL
#define RPL 2     // KV rows per lane per block
#endif
#define WG 64
#define BR (WG * RPL)   // rows per block
#define VPT (DV / WG)   // value dims per lane (8)

REQD_SG64
__kernel void kernel_fa_mla_decode_partial(
        global const char * q_data, ulong q_off, ulong q_nb2,      // q: [DK] f32 per head
        global const char * k_data, ulong k_off, ulong k_nb1,      // K rows (f16)
        global const char * v_data, ulong v_off, ulong v_nb1,      // V rows (f16)
        global const char * m_data, ulong m_off, int has_mask,     // mask row (f16)
        global float * part,                                        // [n_chunks][NH][2 + DV]
        int n_kv, int chunk, float scale)
{
    __local half  q_all[NH * DK];
    __local float p_all[NH * BR];

    const int tid = get_local_id(0);
    const int sg  = get_local_id(1);
    const int c   = get_group_id(0);
    const int h0  = sg * HG;

    for (int i = tid + sg * WG; i < NH * DK; i += WG * (NH / HG)) {
        const int h = i / DK, d = i - h * DK;
        q_all[i] = (half) ((global const float *) (q_data + q_off + h * q_nb2))[d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local const half * q_l = q_all + h0 * DK;
    __local float      * p_l = p_all + h0 * BR;

    float M[HG], L[HG];
    float8 O[HG];
    for (int h = 0; h < HG; h++) { M[h] = -INFINITY; L[h] = 0.0f; O[h] = (float8) 0.0f; }

    const int r_begin = c * chunk;
    const int r_end   = min(n_kv, r_begin + chunk);
    global const half * mask = (global const half *) (m_data + m_off);

    for (int b0 = r_begin; b0 < r_end; b0 += BR) {
        float s[RPL][HG];
        float mv[RPL];
        bool  act[RPL];
        global const half8 * kr[RPL];

        for (int i = 0; i < RPL; i++) {
            const int r = b0 + tid + i * WG;
            const bool valid = r < r_end;
            mv[i]  = (valid && has_mask) ? (float) mask[r] : 0.0f;
            act[i] = valid && mv[i] != -INFINITY;
            kr[i]  = (global const half8 *) (k_data + k_off + (ulong) min(r, r_end - 1) * k_nb1);
            for (int h = 0; h < HG; h++) s[i][h] = 0.0f;
        }

        for (int d8 = 0; d8 < DK / 8; d8++) {
            float8 kv[RPL];
            for (int i = 0; i < RPL; i++) kv[i] = convert_float8(kr[i][d8]);
            for (int h = 0; h < HG; h++) {
                const float8 qv = convert_float8(vload8(0, q_l + h * DK + d8 * 8));
                for (int i = 0; i < RPL; i++) {
                    s[i][h] += dot(kv[i].lo, qv.lo) + dot(kv[i].hi, qv.hi);
                }
            }
        }

        float alpha[HG];
        for (int h = 0; h < HG; h++) {
            float bm = -INFINITY;
            for (int i = 0; i < RPL; i++) {
                s[i][h] = act[i] ? s[i][h] * scale + mv[i] : -INFINITY;
                bm = fmax(bm, s[i][h]);
            }
            bm = sub_group_reduce_max(bm);
            const float mn = fmax(M[h], bm);
            float psum = 0.0f;
            alpha[h] = 1.0f;
            if (mn != -INFINITY) {
                alpha[h] = (M[h] == -INFINITY) ? 0.0f : exp(M[h] - mn);
            }
            for (int i = 0; i < RPL; i++) {
                const float p = (s[i][h] == -INFINITY) ? 0.0f : exp(s[i][h] - mn);
                psum += p;
                p_l[h * BR + i * WG + tid] = p;
            }
            L[h] = L[h] * alpha[h] + sub_group_reduce_add(psum);
            M[h] = mn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int h = 0; h < HG; h++) O[h] *= alpha[h];

        const int nr = min(BR, r_end - b0);
        for (int j = 0; j < nr; j++) {
            const float8 vv = convert_float8(vload8(tid, (global const half *) (v_data + v_off + (ulong) (b0 + j) * v_nb1)));
            for (int h = 0; h < HG; h++) {
                O[h] += p_l[h * BR + j] * vv;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    global float * pc = part + (ulong) c * NH * (2 + DV);
    for (int h = 0; h < HG; h++) {
        global float * ph = pc + (h0 + h) * (2 + DV);
        if (tid == 0) { ph[0] = M[h]; ph[1] = L[h]; }
        vstore8(O[h], 0, ph + 2 + tid * VPT);
    }
}

REQD_SG64
__kernel void kernel_fa_mla_decode_reduce(
        global const float * part, int n_chunks,
        global char * dst, ulong d_off, ulong d_nb1)
{
    const int tid = get_local_id(0);
    const int h   = get_group_id(0);

    float M = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        M = fmax(M, part[((ulong) c * NH + h) * (2 + DV)]);
    }
    float  L = 0.0f;
    float8 O = (float8) 0.0f;
    if (M != -INFINITY) {
        for (int c = 0; c < n_chunks; c++) {
            global const float * ph = part + ((ulong) c * NH + h) * (2 + DV);
            const float mc = ph[0];
            if (mc == -INFINITY) continue;
            const float w = exp(mc - M);
            L += ph[1] * w;
            O += vload8(0, ph + 2 + tid * VPT) * w;
        }
    }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    vstore8(O * inv, 0, (global float *) (dst + d_off + h * d_nb1) + tid * VPT);
}
