#ifndef LDL_LSOLVE
#define LDL_LSOLVE


#include <hls_stream.h>
#include "../xlr8/xlr8_definitions.h"

void ldl_lsolve(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE B[], DTYPE X[]);
void lsolve(spm_info info, hls::stream<int>& jc_stream, hls::stream<int>& ir_stream, hls::stream<DTYPE>& pr_stream, hls::stream<DTYPE>& B_stream, hls::stream<DTYPE>& X_stream, DTYPE* buffer);
   

#endif