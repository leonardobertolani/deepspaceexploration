#ifndef SPARSE_MV
#define SPARSE_MV

#include "../xlr8/xlr8_definitions.h"

#include <hls_stream.h>

void sparse_mv(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag a, flag new_vec);
void spmv_compute_vector(spm_info info, hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& x, hls::stream<DTYPE>& y, DTYPE* result);
void spmv_writeback(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[], flag a, flag new_vec); 

#endif
