#ifndef XLR8_DEFINITIONS
#define XLR8_DEFINITIONS

#include <ap_int.h>
#include <hls_vector.h>

#define MAX_M 569
#define MIN_M 114
#define MAX_N 569
#define MIN_N 223
#define MAX_NNZ 971
#define MIN_NNZ 220
#define MAX_JC (MAX_N + 1)
#define MIN_JC (MIN_N + 1)

#define MAX_IN_VEC 576 //569
#define MIN_IN_VEC 114
#define MAX_OUT_VEC 576

#define INT_BURST 32
#define DTYPE_BURST 16

#define FN_SPMV 0
#define FN_SPMTVM 1
#define FN_LSOLVE 2
#define FN_DSOLVE 3
#define FN_LTSOLVE 4

#define G_BUFF_SIZE 576

typedef double DTYPE;
typedef ap_uint<1> flag;
typedef ap_uint<3> fn_code;
typedef ap_uint<2> params;
typedef struct {
  int n; // columns
  int m; // rows
  int nnz; // non-zero elements
} spm_info;

typedef hls::vector<double, 4> double_vec;
typedef hls::vector<int, 8> int_vec;

typedef struct {
    ap_uint<8> cl;
    ap_uint<8> cl_padded;
} col_len;
typedef hls::vector<int, 4> int4_vec;
typedef hls::vector<double, 8> double8_vec;

#endif
