#ifndef LDL_LTSOLVE_VEC
#define LDL_LTSOLVE_VEC

#include "../xlr8/xlr8_definitions.h"
#include "../xlr8/xlr8_vec.h"

void ldl_ltsolve_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]);
void ltsolve_vec_load_padded(spm_info info, hls::stream<int_vec>& jc, hls::stream<int_vec>& ir, hls::stream<double_vec>& pr, hls::stream<col_len> &cols_lens, hls::stream<double_vec>& pr_padded, hls::stream<int4_vec>& ir_padded);
void ltsolve_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& X, DTYPE* result);
void ltsolve_vec_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]);
void ltsolve_vec_dvec_to_buff(double_vec vec[], DTYPE* buff, size_t len);

#endif
