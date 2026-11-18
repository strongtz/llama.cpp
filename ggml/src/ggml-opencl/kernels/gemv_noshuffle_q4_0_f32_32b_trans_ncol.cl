#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// Multi-column variant of kernel_gemv_noshuffle_q4_0_f32_32b_trans: N = 2..4 activation columns
// (small decode batches, e.g. speculative verification) share one pass over the weights.
// src1: K x N floats (column-major per token, contiguous), dst: M x N floats.

#define QK4_0 32
#define N_SIMDGROUP 4
#ifndef NCOL_MAX
#define NCOL_MAX 4
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_32b_trans_ncol(
        __read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        __read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ncol)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    float sum[NCOL_MAX];
    #pragma unroll
    for (int c = 0; c < NCOL_MAX; ++c) sum[c] = 0.0f;

    for (uint k = groupId; k < (K / QK4_0); k += N_SIMDGROUP) {
        const float scale = (float) src0_d[k * M + gid];
        uint4 regA;
        regA.s0 = read_imageui(src0_q, ((k * 4 + 0) * M + gid)).x;
        regA.s1 = read_imageui(src0_q, ((k * 4 + 1) * M + gid)).x;
        regA.s2 = read_imageui(src0_q, ((k * 4 + 2) * M + gid)).x;
        regA.s3 = read_imageui(src0_q, ((k * 4 + 3) * M + gid)).x;
        const ushort8 b = as_ushort8(regA);
        ushort bb[8] = { b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7 };

        float w[32];
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            w[j] = (float) ((int) ((bb[j >> 2] >> ((j & 3) * 4)) & 0xF) - 8) * scale;
        }

        #pragma unroll
        for (int c = 0; c < NCOL_MAX; ++c) {
            if (c < ncol) {
                float4 y0 = (float4) 0.0f, y1 = (float4) 0.0f;
                if (slid < 4) {
                    const int base = c * (int) (K / 4) + slid * 2 + k * 8;
                    y0 = read_imagef(src1, base);
                    y1 = read_imagef(src1, base + 1);
                }
                float y[8] = { y0.s0, y0.s1, y0.s2, y0.s3, y1.s0, y1.s1, y1.s2, y1.s3 };
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    acc += w[j] * sub_group_broadcast(y[j & 7], j >> 3);
                }
                sum[c] += acc;
            }
        }
    }

    __local float reduceLM[SIMDGROUP_WIDTH * 3 * NCOL_MAX];
    #pragma unroll
    for (int c = 0; c < NCOL_MAX; ++c) {
        if (groupId >= 1) reduceLM[(SIMDGROUP_WIDTH * 3) * c + SIMDGROUP_WIDTH * (groupId - 1) + slid] = sum[c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        dst = (global float *) ((global char *) dst + offsetd);
        #pragma unroll
        for (int c = 0; c < NCOL_MAX; ++c) {
            if (c < ncol) {
                float t = sum[c];
                t += reduceLM[(SIMDGROUP_WIDTH * 3) * c + SIMDGROUP_WIDTH * 0 + slid];
                t += reduceLM[(SIMDGROUP_WIDTH * 3) * c + SIMDGROUP_WIDTH * 1 + slid];
                t += reduceLM[(SIMDGROUP_WIDTH * 3) * c + SIMDGROUP_WIDTH * 2 + slid];
                if (gid < M) {
                    dst[(size_t) c * M + gid] = t;
                }
            }
        }
    }
}
