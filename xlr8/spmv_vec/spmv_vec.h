#ifndef SPMV_VEC
#define SPMV_VEC

#include "../xlr8/xlr8_definitions.h"
#include "../xlr8/xlr8_vec.h"

void spmv_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec x[], double_vec y[], flag a, flag new_vec);
void spmv_vec_xsize(spm_info info, int& x_size);
void spmv_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& x, hls::stream<double_vec>& y, DTYPE* result);
void spmv_vec_writeback(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[], flag a, flag new_vec);

#endif
