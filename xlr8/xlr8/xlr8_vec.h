#ifndef XLR8_VEC
#define XLR8_VEC

#include <hls_stream.h>

#include "xlr8_definitions.h"
#include "../spmv_vec/spmv_vec.h"
#include "../spmtvm_vec/spmtvm_vec.h"
#include "../ldl_lsolve_vec/ldl_lsolve_vec.h"
#include "../ldl_dsolve/ldl_dsolve.h"

void xlr8_read_jc_vec(spm_info info, int_vec jc[], hls::stream<int_vec>& jc_stream);
void xlr8_read_ir_vec(spm_info info, int_vec ir[], hls::stream<int_vec>& ir_stream);
void xlr8_read_pr_vec(spm_info info, double_vec pr[], hls::stream<double_vec>& pr_stream);
void xlr8_read_in_vec(int len, double_vec in[], hls::stream<double_vec>& in_stream);
void xlr8_vec_load_padded(spm_info info, hls::stream<int_vec>& jc, hls::stream<int_vec>& ir, hls::stream<double_vec>& pr, hls::stream<col_len> &cols_lens, hls::stream<double_vec>& pr_padded, hls::stream<int4_vec>& ir_padded);
void xlr8_vec(fn_code function, spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec in_vec[], double_vec out_vec[], params p);

#endif
