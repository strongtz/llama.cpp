#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// A6X compiler incorrectly constant-folds get_local_size() results;
// force runtime materialization via a no-op ALU round-trip.
#ifdef GGML_CL_A6X_CONSTFOLD_FIX
#define MATERIALIZE_WG(x) do { (x) *= 2u; if ((x) > 1u) (x) /= 2u; } while(0)
#else
#define MATERIALIZE_WG(x)
#endif

// assume
#define QK4_0 32
#define N_SIMDGROUP 4

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32(
        __read_only  image1d_buffer_t src0_q,  // quantized A
        global half2  * src0_d,  // A scales
        __read_only  image1d_buffer_t src1,    // B
        ulong offset1,            // offset to B (0)
        global float * dst,     // C
        ulong offsetd,            // offset to C (0)
        int ne00,               // K
        int ne01,               // M
        int ne02,               // 1
        int ne10,               // K
        int ne12,               // 1
        int ne0,                // M
        int ne1,                // N
        int r2,                 // 1
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid    = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A = M / 2;
    // Keep tail lanes active for subgroup ops, but read a valid weight row.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;

    __private uint4     regA;
    __private half2     regS;
    __private float8    regB;

    __private float2 totalSum = (float2)(0.0f);

    uint nsg = get_local_size(1);  // runtime K-split, layout stride stays 4*M
    MATERIALIZE_WG(nsg);

    // loop along K in block granularity, skip nsg blocks every iter
    for (uint k = groupId; k < (K / QK4_0); k += nsg) {
        regS = src0_d[gid_s + k * LINE_STRIDE_A]; // each fiber loads scale of two rows
        // first 4 fibers in each wave load 8 B values to its private scope
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // reduction in local memory over nsg waves (nsg <= 16)
    __local float2 reduceLM[SIMDGROUP_WIDTH * 15];
    if (groupId > 0) reduceLM[SIMDGROUP_WIDTH * (groupId - 1) + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint g = 1; g < nsg; g++) totalSum += reduceLM[SIMDGROUP_WIDTH * (g - 1) + slid];
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the two output rows. The x-grid is padded to CEIL_DIV(ne01/2,64)*64,
        // so when ne01 is not a multiple of 128 the tail row-pairs run past row ne01
        // and would overrun dst into the adjacent tensor. No-op / byte-identical when
        // ne01 % 128 == 0 (M/2 already a multiple of 64 -> no padding).
        if (gid * 2 + 0 < M) dst[gid * 2 + 0] = totalSum.s0;
        if (gid * 2 + 1 < M) dst[gid * 2 + 1] = totalSum.s1;
    }

}

// Multi-column (N in [2..4]) variant of the q4_0 decode GEMV, for the speculative
// / MTP verify batch (n_cols = 2..4 = drafted + bonus positions). Routes the small-
// batch verify OFF the transposed-GEMM dead-zone (gemm_noshuffle_q4_0) onto the
// efficient GEMV path. Each K-block's weights (regA hi+lo) are loaded ONCE and
// reused across the n_cols activation columns. Per-column accumulation is
// independent and identical to n_cols standalone GEMVs. n_cols==3 is byte-identical
// to the original mc3 (col3 disabled, slots 6/7 stay zero). Kept the _mc3 name.
#ifdef VECTOR_SUB_GROUP_BROADCAST
#define MC_DQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define MC_DQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define MC_DQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define MC_DQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif
// One column c: load this column's activation (own brace scope so the macros'
// `shared_y` decl is re-scoped), then dequant (hi+lo) against the shared weights.
#define MC_COL_Q40(ts, c) \
    { if (slid < 4) { regB.s0123 = read_imagef(src1, (c)*COL_STRIDE     + slid*2 + k*8); \
                      regB.s4567 = read_imagef(src1, (c)*COL_STRIDE + 1 + slid*2 + k*8); } \
      MC_DQ_HI(ts, as_ushort8(regA_hi), regS, regB); \
      MC_DQ_LO(ts, as_ushort8(regA_lo), regS, regB); }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_mc3(
        __read_only  image1d_buffer_t src0_q,  // quantized A
        global half2  * src0_d,                // A scales
        __read_only  image1d_buffer_t src1,    // B (n_cols columns, col-major image)
        global float * dst,                    // C (column-major [M x n_cols])
        ulong offsetd,
        int ne00,                              // K
        int ne01,                              // M
        int n_cols)                            // N (2..4)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    // Keep tail lanes active for subgroup ops, but read a valid weight row.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
    // BLOCK_STRIDE_A is the LAYOUT stride between consecutive K-blocks = 4 uints
    // per q4_0 block * M (set by the trans4_ns convert). The "4" is uints/block, NOT
    // the subgroup count — keep it fixed so the K-split count (nsg) can vary.
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;   // = 4 * M (N_SIMDGROUP is the #define 4)
    uint COL_STRIDE     = K / 4;   // float4 pixels per activation column
    uint nsg            = get_local_size(1);  // runtime K-split (4 default, 8 small-M)
    MATERIALIZE_WG(nsg);

    __private uint4  regA_hi, regA_lo;
    __private half2  regS;
    __private float8 regB;

    __private float2 ts0 = (float2)(0.0f);
    __private float2 ts1 = (float2)(0.0f);
    __private float2 ts2 = (float2)(0.0f);
    __private float2 ts3 = (float2)(0.0f);

    for (uint k = groupId; k < (K / QK4_0); k += nsg) {
        regS = src0_d[gid_s + k * LINE_STRIDE_A];

        // weights loaded ONCE, reused across the columns
        regA_hi.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

        MC_COL_Q40(ts0, 0);
        MC_COL_Q40(ts1, 1);
        if (n_cols > 2) MC_COL_Q40(ts2, 2);
        if (n_cols > 3) MC_COL_Q40(ts3, 3);
    }

    // cross-subgroup reduce over nsg subgroups: pack the (up to 4) columns' float2
    // into a float8. Generalized to runtime nsg (4 default, 8 for small-M). Each
    // subgroup writes its partial; subgroup 0 sums the rest into its own acc. At
    // nsg==4 this is byte-identical to the original (sums subgroups 1,2,3 in order).
    __local float8 reduceLM[SIMDGROUP_WIDTH * 8];
    float8 acc = (float8)(ts0.s0, ts0.s1, ts1.s0, ts1.s1, ts2.s0, ts2.s1, ts3.s0, ts3.s1);
    reduceLM[groupId * SIMDGROUP_WIDTH + slid] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        for (uint g = 1; g < nsg; g++) {
            acc += reduceLM[g * SIMDGROUP_WIDTH + slid];
        }
        dst = (global float*)((global char*)dst + offsetd);
        // dst is column-major [M rows x n_cols cols]: (row, col) at col*M + row
        if (gid < LINE_STRIDE_A) {
            vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
            vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
            if (n_cols > 2) vstore2((float2)(acc.s4, acc.s5), 0, &(dst[2 * M + gid * 2]));
            if (n_cols > 3) vstore2((float2)(acc.s6, acc.s7), 0, &(dst[3 * M + gid * 2]));
        }
    }
}
#undef MC_COL_Q40
#undef MC_DQ_HI
#undef MC_DQ_LO


// Multi-column (N = 2..4) q4_0 GEMV with the weights dequantized ONCE per K-block and reused by
// all columns (the _mc3 kernel re-dequantizes per column, so N=2 cost ~2.3x N=1).
inline float8 mcd_dq8(ushort a, ushort b, float s) {
    const ushort8 q = (ushort8)(a & 0xF, (a >> 4) & 0xF, (a >> 8) & 0xF, a >> 12,
                                b & 0xF, (b >> 4) & 0xF, (b >> 8) & 0xF, b >> 12);
    return (convert_float8(q) - 8.0f) * s;
}
inline float mcd_dot8(float8 a, float8 b) {
    return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}
#ifdef VECTOR_SUB_GROUP_BROADCAST
#define MCD_BCAST(v, l) sub_group_broadcast((v), (l))
#else
inline float8 mcd_bcast8(float8 v, uint l) {
    return (float8)(sub_group_broadcast(v.s0, l), sub_group_broadcast(v.s1, l), sub_group_broadcast(v.s2, l), sub_group_broadcast(v.s3, l),
                    sub_group_broadcast(v.s4, l), sub_group_broadcast(v.s5, l), sub_group_broadcast(v.s6, l), sub_group_broadcast(v.s7, l));
}
#define MCD_BCAST(v, l) mcd_bcast8((v), (l))
#endif

#define MCD_PART(bits, l0, w00, w01, w10, w11) \
    w00 = mcd_dq8(bits.s0, bits.s2, s0); \
    w01 = mcd_dq8(bits.s4, bits.s6, s0); \
    w10 = mcd_dq8(bits.s1, bits.s3, s1); \
    w11 = mcd_dq8(bits.s5, bits.s7, s1);

#ifdef MCD_UNIFORM_LOADS
// every lane reads the (same) activations directly: uniform image reads instead of subgroup broadcasts
#define MCD_COL(acc, yb, l0) { \
    const uint b_ = (uint) (yb) * COL_STRIDE + k * 8 + (l0) * 2; \
    const float8 ya_ = (float8)(read_imagef(src1, b_ + 0), read_imagef(src1, b_ + 1)); \
    const float8 yc_ = (float8)(read_imagef(src1, b_ + 2), read_imagef(src1, b_ + 3)); \
    acc.s0 += mcd_dot8(w00, ya_) + mcd_dot8(w01, yc_); \
    acc.s1 += mcd_dot8(w10, ya_) + mcd_dot8(w11, yc_); }
#else
#define MCD_COL(acc, yb, l0) { \
    const float8 ya_ = MCD_BCAST(yb, (l0)); \
    const float8 yc_ = MCD_BCAST(yb, (l0) + 1); \
    acc.s0 += mcd_dot8(w00, ya_) + mcd_dot8(w01, yc_); \
    acc.s1 += mcd_dot8(w10, ya_) + mcd_dot8(w11, yc_); }
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_mcd(
        __read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        __read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_cols)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;
    uint COL_STRIDE     = K / 4;
    uint nsg            = get_local_size(1);
    MATERIALIZE_WG(nsg);

    float2 a0 = (float2)(0.0f), a1 = (float2)(0.0f), a2 = (float2)(0.0f), a3 = (float2)(0.0f);

    for (uint k = groupId; k < (K / QK4_0); k += nsg) {
        const half2 regS = src0_d[gid_s + k * LINE_STRIDE_A];
        const float s0 = (float) regS.s0, s1 = (float) regS.s1;

        uint4 rh, rl;
        rh.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        rh.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        rh.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        rh.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        rl.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        rl.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        rl.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        rl.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

#ifndef MCD_UNIFORM_LOADS
        float8 y0 = (float8)(0.0f), y1 = (float8)(0.0f), y2 = (float8)(0.0f), y3 = (float8)(0.0f);
        if (slid < 4) {
            y0 = (float8)(read_imagef(src1, 0 * COL_STRIDE + slid * 2 + k * 8), read_imagef(src1, 0 * COL_STRIDE + 1 + slid * 2 + k * 8));
            y1 = (float8)(read_imagef(src1, 1 * COL_STRIDE + slid * 2 + k * 8), read_imagef(src1, 1 * COL_STRIDE + 1 + slid * 2 + k * 8));
            if (n_cols > 2) y2 = (float8)(read_imagef(src1, 2 * COL_STRIDE + slid * 2 + k * 8), read_imagef(src1, 2 * COL_STRIDE + 1 + slid * 2 + k * 8));
            if (n_cols > 3) y3 = (float8)(read_imagef(src1, 3 * COL_STRIDE + slid * 2 + k * 8), read_imagef(src1, 3 * COL_STRIDE + 1 + slid * 2 + k * 8));
        }
#endif

        float8 w00, w01, w10, w11;
#ifdef MCD_UNIFORM_LOADS
#define Y0 0
#define Y1 1
#define Y2 2
#define Y3 3
#else
#define Y0 y0
#define Y1 y1
#define Y2 y2
#define Y3 y3
#endif
        const ushort8 bh = as_ushort8(rh);
        MCD_PART(bh, 0, w00, w01, w10, w11);
        MCD_COL(a0, Y0, 0);
        MCD_COL(a1, Y1, 0);
        if (n_cols > 2) MCD_COL(a2, Y2, 0);
        if (n_cols > 3) MCD_COL(a3, Y3, 0);

        const ushort8 bl = as_ushort8(rl);
        MCD_PART(bl, 2, w00, w01, w10, w11);
        MCD_COL(a0, Y0, 2);
        MCD_COL(a1, Y1, 2);
        if (n_cols > 2) MCD_COL(a2, Y2, 2);
        if (n_cols > 3) MCD_COL(a3, Y3, 2);
#undef Y0
#undef Y1
#undef Y2
#undef Y3
    }

    __local float8 reduceLM_mcd[SIMDGROUP_WIDTH * 8];
    float8 acc = (float8)(a0.s0, a0.s1, a1.s0, a1.s1, a2.s0, a2.s1, a3.s0, a3.s1);
    reduceLM_mcd[groupId * SIMDGROUP_WIDTH + slid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint g = 1; g < nsg; g++) {
            acc += reduceLM_mcd[g * SIMDGROUP_WIDTH + slid];
        }
        dst = (global float *) ((global char *) dst + offsetd);
        if (gid < LINE_STRIDE_A) {
            vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
            vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
            if (n_cols > 2) vstore2((float2)(acc.s4, acc.s5), 0, &(dst[2 * M + gid * 2]));
            if (n_cols > 3) vstore2((float2)(acc.s6, acc.s7), 0, &(dst[3 * M + gid * 2]));
        }
    }
}
#undef MCD_PART
#undef MCD_COL
#undef MCD_BCAST


// fp16-math multi-column q4_0 GEMV (N = 2..4): weights dequantized once per block into half8, activations
// pre-converted to half, 16-product partial dots in half accumulated into float. Adreno 6xx runs fp16
// at twice the fp32 rate; the q4_0 decode GEMVs are ALU-bound, so this is what makes N > 1 cheap.
__kernel void kernel_mch_cvt_act_f16(__global const char * src, ulong offs, __global half * dst, int n) {
    const int i = get_global_id(0);
    if (i < n) {
        dst[i] = (half) ((__global const float *) (src + offs))[i];
    }
}

inline half8 mch_dq8(ushort a, ushort b, half s) {
    const ushort8 q = (ushort8)(a & 0xF, (a >> 4) & 0xF, (a >> 8) & 0xF, a >> 12,
                                b & 0xF, (b >> 4) & 0xF, (b >> 8) & 0xF, b >> 12);
    return (convert_half8(q) - (half) 8.0h) * s;
}
inline half mch_dot8(half8 a, half8 b) {
    return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

#define MCH_COL(acc, c, l0) { \
    const uint b_ = (uint) (c) * COL_STRIDE + k * 8 + (l0) * 2; \
    const half8 ya_ = (half8)(read_imageh(src1, b_ + 0), read_imageh(src1, b_ + 1)); \
    const half8 yc_ = (half8)(read_imageh(src1, b_ + 2), read_imageh(src1, b_ + 3)); \
    acc.s0 += (float) (mch_dot8(w00, ya_) + mch_dot8(w01, yc_)); \
    acc.s1 += (float) (mch_dot8(w10, ya_) + mch_dot8(w11, yc_)); }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_mch(
        __read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        __read_only  image1d_buffer_t src1,     // half activations, n_cols x K
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_cols)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;
    uint COL_STRIDE     = K / 4;
    uint nsg            = get_local_size(1);
    MATERIALIZE_WG(nsg);

    float2 a0 = (float2)(0.0f), a1 = (float2)(0.0f), a2 = (float2)(0.0f), a3 = (float2)(0.0f);

    for (uint k = groupId; k < (K / QK4_0); k += nsg) {
        const half2 s = src0_d[gid_s + k * LINE_STRIDE_A];
        const half s0 = s.s0, s1 = s.s1;

        uint4 rh, rl;
        rh.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        rh.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        rh.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        rh.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        rl.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        rl.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        rl.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        rl.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

        half8 w00, w01, w10, w11;
        const ushort8 bh = as_ushort8(rh);
        w00 = mch_dq8(bh.s0, bh.s2, s0); w01 = mch_dq8(bh.s4, bh.s6, s0);
        w10 = mch_dq8(bh.s1, bh.s3, s1); w11 = mch_dq8(bh.s5, bh.s7, s1);
        MCH_COL(a0, 0, 0);
        if (n_cols > 1) MCH_COL(a1, 1, 0);
        if (n_cols > 2) MCH_COL(a2, 2, 0);
        if (n_cols > 3) MCH_COL(a3, 3, 0);

        const ushort8 bl = as_ushort8(rl);
        w00 = mch_dq8(bl.s0, bl.s2, s0); w01 = mch_dq8(bl.s4, bl.s6, s0);
        w10 = mch_dq8(bl.s1, bl.s3, s1); w11 = mch_dq8(bl.s5, bl.s7, s1);
        MCH_COL(a0, 0, 2);
        if (n_cols > 1) MCH_COL(a1, 1, 2);
        if (n_cols > 2) MCH_COL(a2, 2, 2);
        if (n_cols > 3) MCH_COL(a3, 3, 2);
    }

    __local float8 reduceLM_mch[SIMDGROUP_WIDTH * 8];
    float8 acc = (float8)(a0.s0, a0.s1, a1.s0, a1.s1, a2.s0, a2.s1, a3.s0, a3.s1);
    reduceLM_mch[groupId * SIMDGROUP_WIDTH + slid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint g = 1; g < nsg; g++) {
            acc += reduceLM_mch[g * SIMDGROUP_WIDTH + slid];
        }
        dst = (global float *) ((global char *) dst + offsetd);
        if (gid < LINE_STRIDE_A) {
            vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
            if (n_cols > 1) vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
            if (n_cols > 2) vstore2((float2)(acc.s4, acc.s5), 0, &(dst[2 * M + gid * 2]));
            if (n_cols > 3) vstore2((float2)(acc.s6, acc.s7), 0, &(dst[3 * M + gid * 2]));
        }
    }
}
#undef MCH_COL
