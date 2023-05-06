#ifndef CSR5_SPMV_AVX2_H
#define CSR5_SPMV_AVX2_H

#include "common_avx2.h"
#include "utils_avx2.h"

#ifndef __cplusplus
#include <stdalign.h>   // C11 defines _Alignas().  This header defines alignas()
#endif

template<typename T>
void print_bit(T in){
    for (long long k = sizeof(T) * 8 - 1; k >= 0; k--) {
        printf("%d", (in >> k) & 1);
        if (k % 8 == 0)
            printf(" ");
    }
    printf("\n");
}

void p128_hex_u8(__m128i in) {
    alignas(16) uint8_t v[16];
    _mm_store_si128((__m128i*)v, in);
    printf("v16_u8: %x %x %x %x | %x %x %x %x | %x %x %x %x | %x %x %x %x\n",
           v[0], v[1],  v[2],  v[3],  v[4],  v[5],  v[6],  v[7],
           v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}

void p128_hex_u16(__m128i in) {
    alignas(16) uint16_t v[8];
    _mm_store_si128((__m128i*)v, in);
    printf("v8_u16: %x %x %x %x,  %x %x %x %x\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

void p128_hex_u32(__m128i in) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, in);
    printf("v4_u32: %x %x %x %x\n", v[0], v[1], v[2], v[3]);
}

void p128_bin_u32(__m128i in) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, in);
    for(int i = 0; i < 4; i ++ ){
        print_bit<unsigned int>(v[i]);
    }
}

void p128_hex_u64(__m128i in) {
    alignas(16) unsigned long long v[2];  // uint64_t might give format-string warnings with %llx; it's just long in some ABIs
    _mm_store_si128((__m128i*)v, in);
    printf("v2_u64: %llx %llx\n", v[0], v[1]);
}

void p256_bin_u64(__m256i in) {
    alignas(16) unsigned long long v[4];  // uint64_t might give format-string warnings with %llx; it's just long in some ABIs
    _mm256_store_si256((__m256i*)v, in);
    for(int i = 0; i < 4; i ++ ){
        print_bit<unsigned long long>(v[i]);
    }
}

void p256_bin_i64(__m256i in) {
    alignas(16) long long v[4];  // uint64_t might give format-string warnings with %llx; it's just long in some ABIs
    _mm256_store_si256((__m256i*)v, in);
    for(int i = 0; i < 4; i ++ ){
        print_bit<long long>(v[i]);
    }
}

void p256_d64(__m256d in) {
    double v[4];
    _mm256_store_pd(v, in);
    printf("v4_d64: %.2lf %.2lf %.2lf %.2lf\n", v[0], v[1], v[2], v[3]);
}

void p128_dec_i32(__m128i in) {
    alignas(16) int32_t v[4];
    _mm_store_si128((__m128i*)v, in);
    printf("v2_u64: %d %d %d %d\n", v[0], v[1], v[2], v[3]);
}


template<typename iT, typename vT>
inline void partition_fast_track(const vT           *d_value_partition,
                                 const vT           *d_x,
                                 const iT           *d_column_index_partition,
                                 vT                 *d_calibrator,
                                 vT                 *d_y,
                                 const iT            row_start,
                                 const iT            par_id,
                                 const int           tid,
                                 const iT            start_row_start,
                                 const vT            alpha,
                                 const int           sigma,
                                 const int           stride_vT,
                                 const bool          direct)
{
    __m256d sum256d   = _mm256_setzero_pd();
    __m256d value256d, x256d;
    vT x256d0, x256d1, x256d2, x256d3;

    #pragma unroll(ANONYMOUSLIB_CSR5_SIGMA)
    for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        value256d = _mm256_load_pd(&d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);

        x256d0 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]];
        x256d1 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 1]];
        x256d2 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 2]];
        x256d3 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 3]];
        x256d = _mm256_set_pd(x256d3, x256d2, x256d1, x256d0);

        sum256d = _mm256_fmadd_pd(value256d, x256d, sum256d);
    }

    vT sum = hsum_avx(sum256d);

    if (row_start == start_row_start && !direct)
        d_calibrator[tid * stride_vT] += sum;
    else{
        if(direct)
            d_y[row_start] = sum;
        else
            d_y[row_start] += sum;
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_compute_kernel(const iT           *d_column_index,
                              const vT           *d_value,
                              const iT           *d_row_pointer,
                              const vT           *d_x,
                              const uiT          *d_partition_pointer,
                              const uiT          *d_partition_descriptor,
                              const iT           *d_partition_descriptor_offset_pointer,
                              const iT           *d_partition_descriptor_offset,
                              vT                 *d_calibrator,
                              vT                 *d_y,
                              const iT            p,
                              const int           num_packet,
                              const int           bit_y_offset,
                              const int           bit_scansum_offset,
                              const vT            alpha,
                              const int           c_sigma)
{
    const int num_thread = omp_get_max_threads();
    const int chunk = ceil((double)(p-1) / (double)num_thread);
    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    const int num_thread_active = ceil((p-1.0)/chunk);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        iT start_row_start = tid < num_thread_active ? d_partition_pointer[tid * chunk] & 0x7FFFFFFF : 0;

        vT  s_sum[8]; // allocate a cache line
        vT  s_first_sum[8]; // allocate a cache line
        uint64_t s_cond[8]; // allocate a cache line
        int s_y_idx[16]; // allocate a cache line

        int inc0, inc1, inc2, inc3;
        vT x256d0, x256d1, x256d2, x256d3;

        __m128i *d_column_index_partition128i;
        __m128i *d_partition_descriptor128i;

        __m256d sum256d = _mm256_setzero_pd();
        __m256d tmp_sum256d = _mm256_setzero_pd();
        __m256d first_sum256d = _mm256_setzero_pd();
        __m256d last_sum256d = _mm256_setzero_pd();
        __m128i scansum_offset128i, y_offset128i, y_idx128i;
        __m256i start256i;
        __m256i stop256i = _mm256_setzero_si256();

        __m256d value256d, x256d;

        __m256i local_bit256i;
        __m256i direct256i;

        __m128i descriptor128i;
        __m256i tmp256i;
        // #pragma omp for schedule(static, chunk)
        for (int par_id = 0; par_id < p - 1; par_id++)
        {
            printf("\n! ======== tile %d ========\n", par_id);
            const iT *d_column_index_partition = &d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];
            const vT *d_value_partition = &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];

            uiT row_start     = d_partition_pointer[par_id];
            const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

            if (row_start == row_stop) // fast track through reduction
            {
                // check whether the the partition contains the first element of row "row_start"
                // => we are the first writing data to d_y[row_start]
                bool fast_direct = (d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet] >>
                                                    (31 - (bit_y_offset + bit_scansum_offset)) & 0x1);
                partition_fast_track<iT, vT>
                        (d_value_partition, d_x, d_column_index_partition,
                         d_calibrator, d_y, row_start, par_id, tid, start_row_start, alpha, c_sigma, stride_vT, fast_direct);
            }
            else // normal track for all the other partitions
            {
                const bool empty_rows = (row_start >> 31) & 0x1;
                row_start &= 0x7FFFFFFF;

                vT *d_y_local = &d_y[row_start+1];
                const int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

                d_column_index_partition128i = (__m128i *)d_column_index_partition;
                d_partition_descriptor128i = (__m128i *)&d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet];

                first_sum256d = _mm256_setzero_pd();
                stop256i = _mm256_setzero_si256();

                descriptor128i = _mm_load_si128(d_partition_descriptor128i);
                printf("\ndescriptor128i original\n");
                p128_bin_u32(descriptor128i);

                // y_offset128i和scansum_offset128i分别只保留了y_offset和seg_offset的部分，原本是保存在一个packet的32位整数中的
                y_offset128i = _mm_srli_epi32(descriptor128i, 32 - bit_y_offset);
                scansum_offset128i = _mm_slli_epi32(descriptor128i, bit_y_offset);
                scansum_offset128i = _mm_srli_epi32(scansum_offset128i, 32 - bit_scansum_offset);

                printf("\ny_offset128i\n");
                p128_dec_i32(y_offset128i);
                
                printf("\nscansum_offset128i\n");
                p128_bin_u32(scansum_offset128i);

                // descriptor128i左移, 只保留bit_flag, 且从高位开始存储
                descriptor128i = _mm_slli_epi32(descriptor128i, bit_y_offset + bit_scansum_offset);
                printf("\ndescriptor128i only bit flag\n");
                p128_bin_u32(descriptor128i);
                
                // remember if the first element of this partition is the first element of a new row
                // descriptor128i右移31位，保留了bit_flag的首位，即tile第一行的bit_flag的值, 即bit_flag[0, :]
                // 这个local_bit只是用来计算了first_direct，后就没用了
                local_bit256i = _mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31));
                printf("\nlocal_bit256i\n");
                p256_bin_i64(local_bit256i);
                bool first_direct = false;
                _mm256_store_si256((__m256i *)s_cond, local_bit256i);
                // s_cond[0] = bit_flag[0, 0]
                if(s_cond[0])
                    first_direct = true;
                printf("\nfirst direct: %d\n", first_direct);
                    
                // remember if the first element of the first partition of the current thread is the first element of a new row
                bool first_all_direct = false;
                if(par_id == tid * chunk)
                    first_all_direct = first_direct;
                printf("\nfirst_all_direct direct: %d\n", first_all_direct);
                        
                // bit_flag[0, 0] = True 
                descriptor128i = _mm_or_si128(descriptor128i, _mm_set_epi32(0, 0, 0, 0x80000000));
                printf("\ndescriptor128i set first element of every tile to be True\n");
                p128_bin_u32(descriptor128i);

                // local_bit256i = bit_flag[0, :], tile第一行的bit_flag的值
                local_bit256i = _mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31));  
                // bit_flag[0, :]只保留了bit_flag每个lane的第一个值, 且每个tile的第一个bit已经为True
                // start256i = 1 - bit_flag[0, :]
                start256i = _mm256_sub_epi64(_mm256_set1_epi64x(0x1), local_bit256i);       
                // direct256i[0] = bit_flag[0, 0] && 0 = 0， 再设置为false
                // direct256i[1:] = bit_flag[0, 1:] && 1
                direct256i = _mm256_and_si256(local_bit256i, _mm256_set_epi64x(0x1, 0x1, 0x1, 0)); // 再设置为false

                printf("\nlocal_bit256i after set True\n");
                p256_bin_i64(local_bit256i);
                printf("\nstart256i\n");
                p256_bin_i64(start256i);
                printf("\ndirect256i\n");
                p256_bin_i64(direct256i);

                printf("\n! ----- row 0 of All simga rows ------\n");
                value256d = _mm256_load_pd(d_value_partition);

                printf("\nvalue256d\n");
                p256_d64(value256d);

                x256d0 = d_x[d_column_index_partition[0]];
                x256d1 = d_x[d_column_index_partition[1]];
                x256d2 = d_x[d_column_index_partition[2]];
                x256d3 = d_x[d_column_index_partition[3]];
                x256d = _mm256_set_pd(x256d3, x256d2, x256d1, x256d0);

                printf("\nx256d\n");
                p256_d64(x256d);

                // 已经计算了每一行的4个lane
                sum256d = _mm256_mul_pd(value256d, x256d);
        
                printf("\nsum256d\n");
                p256_d64(sum256d);

                // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 23
                int ly = 0;
#endif

                for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
                {
                    printf("\n! ----- row %d of All simga rows ------\n", i);
                    x256d0 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]];
                    x256d1 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 1]];
                    x256d2 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 2]];
                    x256d3 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 3]];
                    x256d = _mm256_set_pd(x256d3, x256d2, x256d1, x256d0);

                    printf("\nx256d\n");
                    p256_d64(x256d);

#if ANONYMOUSLIB_CSR5_SIGMA > 23
                    int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

                    if (!(ly || norm_i) || (ly && !(norm_i % 32)))
                    {
                        ly++;
                        descriptor128i = _mm_load_si128(&d_partition_descriptor128i[ly]);
                    }
                    norm_i = !ly ? i : norm_i;
                    norm_i = 31 - norm_i % 32;

                    local_bit256i = _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, norm_i)), _mm256_set1_epi64x(0x1));
#else
                    // local_bit256i保存了4个lane的bit_flag，local_bit256i = bit_flag[i, :]
                    local_bit256i = _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31-i)), _mm256_set1_epi64x(0x1));
#endif
                    printf("\n!calculating store_to_offchip");
                    printf("\nlocal_bit256i\n");
                    p256_bin_i64(local_bit256i);
                    // 按位与，结果为0则返回1, 因此local_bit256i全为0，则store_to_chip = 1
                    int store_to_offchip = _mm256_testz_si256(local_bit256i, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF));
                    printf("\nstore_to_offchip: %d\n", store_to_offchip);
                    if (!store_to_offchip)
                    {

                        // 要根据y_offset[i] 或者 empty_offset[y_offset[i]]来计算
                        y_idx128i = empty_rows ? _mm_i32gather_epi32 (&d_partition_descriptor_offset[offset_pointer], y_offset128i, 4) : y_offset128i;
                        
                        printf("\ny_idx128i\n");
                        p128_dec_i32(y_idx128i);

                        // mask scatter store
                        _mm_store_si128((__m128i *)s_y_idx, y_idx128i);
                        printf("\ns_y_idx: %d %d %d %d\n", s_y_idx[0], s_y_idx[1], s_y_idx[2], s_y_idx[3]);

                        _mm256_store_pd(s_sum, sum256d);
                        printf("\ns_sum: %.2lf %.2lf %.2lf %.2lf\n", s_sum[0], s_sum[1], s_sum[2], s_sum[3]);

                        // green的结尾, 当前bit_flag为true，表明是新的一行的开始，要把之前的sum写回，direct256i为True，表明是这一列之前出现过True，中间的这一段就是Green
                        _mm256_store_si256((__m256i *)s_cond, _mm256_and_si256(direct256i, local_bit256i));
                        
                        printf("\ns_cond: %lld %lld %lld %lld\n", s_cond[0], s_cond[1], s_cond[2], s_cond[3]);

                        inc0 = 0, inc1 = 0, inc2 = 0, inc3 = 0;
                        // 直接写回，并将y_offset + 1
                        if (s_cond[0]) {d_y_local[s_y_idx[0]] = s_sum[0]; inc0 = 1;}
                        if (s_cond[1]) {d_y_local[s_y_idx[1]] = s_sum[1]; inc1 = 1;}
                        if (s_cond[2]) {d_y_local[s_y_idx[2]] = s_sum[2]; inc2 = 1;}
                        if (s_cond[3]) {d_y_local[s_y_idx[3]] = s_sum[3]; inc3 = 1;}

                        y_offset128i = _mm_add_epi32(y_offset128i, _mm_set_epi32(inc3, inc2, inc1, inc0));
                        printf("\ny_offset128i after modify\n");
                        p128_dec_i32(y_offset128i);

                        
                        // tmp256i = (not (direct256i == 1)) and (local_bit256i == 1) = 
                        tmp256i = _mm256_andnot_si256(
                                    _mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1)),
                                    _mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0x1)));

                        printf("\ntmp256i tmp256i\n");
                        p256_bin_i64(tmp256i);

                        printf("\nfirst_sum256d before\n");
                        p256_d64(first_sum256d);

                        printf("\nsum256d before\n");
                        p256_d64(sum256d);

                        // if(tmp256i == 0) first_sum256d = first_sum256d
                        // if(tmp256i == 1) first_sum256d = sum256d
                        first_sum256d = _mm256_add_pd(
                                    _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0))),   first_sum256d),
                                    _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF))), sum256d));
                        printf("\nfirst_sum256d after\n");
                        p256_d64(first_sum256d);

                        sum256d = _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0))), sum256d);
                        printf("\nsum256d after\n");
                        p256_d64(sum256d);

                        printf("\ndirect256i before\n");
                        p256_bin_i64(direct256i);

                        direct256i = _mm256_or_si256(direct256i, local_bit256i);
                        printf("\ndirect256i after\n");
                        p256_bin_i64(direct256i);

                        stop256i = _mm256_add_epi64(stop256i, local_bit256i);
                    }

                    value256d = _mm256_load_pd(&d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
                    sum256d = _mm256_fmadd_pd(value256d, x256d, sum256d);
                }

                tmp256i = _mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1));

                first_sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);
                tmp256i = _mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0));
                first_sum256d = _mm256_add_pd(first_sum256d, _mm256_and_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                last_sum256d = sum256d;

                tmp256i = _mm256_cmpeq_epi64(start256i, _mm256_set1_epi64x(0x1));
                sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);

                sum256d = _mm256_permute4x64_pd(sum256d, 0x39);
                sum256d = _mm256_and_pd(_mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000)), sum256d);

                tmp_sum256d = sum256d;
                sum256d = hscan_avx(sum256d);

                scansum_offset128i = _mm_add_epi32(scansum_offset128i, _mm_set_epi32(3, 2, 1, 0));

                tmp256i = _mm256_castsi128_si256(scansum_offset128i);
                tmp256i = _mm256_permutevar8x32_epi32(tmp256i, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
                tmp256i = _mm256_add_epi32(tmp256i, tmp256i);
                tmp256i = _mm256_add_epi32(tmp256i, _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0));

                sum256d = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(sum256d), tmp256i)), sum256d);
                sum256d = _mm256_add_pd(sum256d, tmp_sum256d);

                tmp256i = _mm256_cmpgt_epi64(start256i, stop256i);
                tmp256i = _mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0));

                last_sum256d = _mm256_add_pd(last_sum256d, _mm256_and_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                y_idx128i = empty_rows ? _mm_i32gather_epi32 (&d_partition_descriptor_offset[offset_pointer], y_offset128i, 4) : y_offset128i;

                _mm256_store_si256((__m256i *)s_cond, direct256i);
                _mm_store_si128((__m128i *)s_y_idx, y_idx128i);
                _mm256_store_pd(s_sum, last_sum256d);

                if (s_cond[0]) {d_y_local[s_y_idx[0]] = s_sum[0]; _mm256_store_pd(s_first_sum, first_sum256d);}
                if (s_cond[1]) d_y_local[s_y_idx[1]] = s_sum[1];
                if (s_cond[2]) d_y_local[s_y_idx[2]] = s_sum[2];
                if (s_cond[3]) d_y_local[s_y_idx[3]] = s_sum[3];

                // only use calibrator if this partition does not contain the first element of the row "row_start"
                if (row_start == start_row_start && !first_all_direct)
                    d_calibrator[tid * stride_vT] += s_cond[0] ? s_first_sum[0] : s_sum[0];
                else{
                    if(first_direct)
                        d_y[row_start] = s_cond[0] ? s_first_sum[0] : s_sum[0];
                    else
                        d_y[row_start] += s_cond[0] ? s_first_sum[0] : s_sum[0];
                }
            }
        }
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                vT        *d_calibrator,
                                vT        *d_y,
                                const iT   p)
{
    int num_thread = omp_get_max_threads();
    int chunk = ceil((double)(p-1) / (double)num_thread);
    int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    // calculate the number of maximal active threads (for a static loop scheduling with size chunk)
    int num_thread_active = ceil((p-1.0)/chunk);
    int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

    for (int i = 0; i < num_cali; i++)
    {
        d_y[(d_partition_pointer[i * chunk] << 1) >> 1] += d_calibrator[i * stride_vT];
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_tail_partition_kernel(const iT           *d_row_pointer,
                                     const iT           *d_column_index,
                                     const vT           *d_value,
                                     const vT           *d_x,
                                     vT                 *d_y,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const iT            m,
                                     const int           sigma,
                                     const vT            alpha)
{
    const iT index_first_element_tail = (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma;
    
    // #pragma omp parallel for
    for (iT row_id = tail_partition_start; row_id < m; row_id++)
    {
        const iT idx_start = row_id == tail_partition_start ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
        const iT idx_stop  = d_row_pointer[row_id + 1];

        vT sum = 0;
        for (iT idx = idx_start; idx < idx_stop; idx++)
            sum += d_value[idx] * d_x[d_column_index[idx]];// * alpha;

        if(row_id == tail_partition_start && d_row_pointer[row_id] != index_first_element_tail){
            d_y[row_id] = d_y[row_id] + sum;
        }else{
            d_y[row_id] = sum;
        }
    }
}


template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int csr5_spmv(const int                 sigma,
              const ANONYMOUSLIB_IT         p,
              const ANONYMOUSLIB_IT         m,
              const int                 bit_y_offset,
              const int                 bit_scansum_offset,
              const int                 num_packet,
              const ANONYMOUSLIB_IT        *row_pointer,
              const ANONYMOUSLIB_IT        *column_index,
              const ANONYMOUSLIB_VT        *value,
              const ANONYMOUSLIB_UIT       *partition_pointer,
              const ANONYMOUSLIB_UIT       *partition_descriptor,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset_pointer,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset,
              ANONYMOUSLIB_VT              *calibrator,
              const ANONYMOUSLIB_IT         tail_partition_start,
              const ANONYMOUSLIB_VT         alpha,
              const ANONYMOUSLIB_VT        *x,
              ANONYMOUSLIB_VT              *y)
{
    int err = ANONYMOUSLIB_SUCCESS;

    const int num_thread = omp_get_max_threads();
    memset(calibrator,0,ANONYMOUSLIB_X86_CACHELINE*num_thread);

    printf("\nspmv_csr5_compute_kernel\n");
    spmv_csr5_compute_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (column_index, value, row_pointer, x,
             partition_pointer, partition_descriptor,
             partition_descriptor_offset_pointer, partition_descriptor_offset,
             calibrator, y, p,
             num_packet, bit_y_offset, bit_scansum_offset, alpha, sigma);

    printf("\nspmv_csr5_calibrate_kernel\n");
    spmv_csr5_calibrate_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (partition_pointer, calibrator, y, p);

    printf("\nspmv_csr5_tail_partition_kernel\n");
    spmv_csr5_tail_partition_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (row_pointer, column_index, value, x, y,
             tail_partition_start, p, m, sigma, alpha);

    return err;
}

#endif // CSR5_SPMV_AVX2_H
