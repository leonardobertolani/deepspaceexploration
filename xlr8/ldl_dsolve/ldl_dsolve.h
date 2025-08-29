#ifndef LDL_DSOLVE
#define LDL_DSOLVE

#include <hls_stream.h>
#include "../xlr8/xlr8_definitions.h"

void ldl_dsolve(spm_info info, double_vec pr[], double_vec X[], double_vec B[]);
void dsolve(spm_info info, hls::stream<double_vec>& pr, hls::stream<double_vec>& B_stream, hls::stream<double_vec>& X_stream);
void d_copy_streams(spm_info info, double_vec pr[], double_vec B[], hls::stream<double_vec>& pr_stream, hls::stream<double_vec>& B_stream);
void d_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]);

#endif