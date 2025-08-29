#ifndef LDL_LSOLVE_VEC
#define LDL_LSOLVE_VEC

#include "../xlr8/xlr8_definitions.h"
#include "../xlr8/xlr8_vec.h"

void ldl_lsolve_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]);
void lsolve_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& X, DTYPE* result);
void lsolve_vec_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]);
void lsolve_dvec_to_buff(double_vec vec[], DTYPE* buff, size_t len);

#endif
