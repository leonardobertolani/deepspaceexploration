#include <stdio.h>
#include "ldl_dsolve.h"

void d_copy_streams(spm_info info, double_vec pr[], double_vec B[], hls::stream<double_vec>& pr_stream, hls::stream<double_vec>& B_stream) {
    double_vec v_pr, v_B;
    int vector_size = (info.nnz + (4 - info.nnz%4)) >> 2;
    for (int i = 0; i < 143; i++) {
        #pragma HLS PIPELINE
        v_pr = pr[i];
        v_B = B[i];
        pr_stream.write(v_pr);
        B_stream.write(v_B);
    }
}

void dsolve(spm_info info, hls::stream<double_vec>& pr, hls::stream<double_vec>& B_stream, hls::stream<double_vec>& X_stream) {
    #pragma HLS ALLOCATION operation instances=ddiv limit=1
    double_vec bi, pri, div;
    int vector_size = (info.nnz + (4 - info.nnz%4)) >> 2;
    WRITE_RESULT: for (int i = 0; i < 143; i++) {
        #pragma HLS PIPELINE

        bi = B_stream.read();
        pri = pr.read();
        div = bi / pri;
        X_stream.write(div);
    }
}

void d_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]) {
    double_vec v;
    int vector_size = (info.nnz + (4 - info.nnz%4)) >> 2;
    for (int i = 0; i < 143; i++) {
        #pragma HLS PIPELINE
        v = X_stream.read();
        X[i] = v;
    }
}

void ldl_dsolve(spm_info info, double_vec pr[], double_vec X[], double_vec B[]) {
    #pragma HLS INTERFACE mode=m_axi bundle=gmem2 depth=MAX_NNZ port=pr offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=gmem3 depth=MAX_OUT_VEC port=X offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=gmem4 depth=MAX_IN_VEC port=B offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=256

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=X bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    hls::stream<double_vec> pr_stream, X_stream, B_stream;

    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ
    #pragma HLS STREAM variable=X_stream  depth=MAX_OUT_VEC
    #pragma HLS STREAM variable=B_stream  depth=MAX_IN_VEC

    d_copy_streams(info, pr, B, pr_stream, B_stream);
    dsolve(info, pr_stream, B_stream, X_stream);
    d_writeback(info, X_stream, X);
}