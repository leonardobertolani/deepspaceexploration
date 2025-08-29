#ifndef SPARSE_MTVM
#define SPARSE_MTVM

#include "../xlr8/xlr8_definitions.h"

#include <hls_stream.h>

void sparse_mtvm(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag new_vec, flag skip_diagonal);
void spmtv_compute(spm_info info, hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& x, hls::stream<DTYPE>& y, flag skip_diagonal, DTYPE* gbuff);
void spmtv_writeback(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[], flag new_vec);

#endif
