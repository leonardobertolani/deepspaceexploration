#ifndef XLR8
#define XLR8

#include <hls_stream.h>

#include "xlr8_definitions.h"

void xlr8(fn_code function, spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE in_vec[], DTYPE out_vec[], params p);

#endif
