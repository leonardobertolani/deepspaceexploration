#include "xlr8.h"
#include "../sparse_mv/sparse_mv.h"
#include "../sparse_mtvm/sparse_mtvm.h"
#include "../ldl_lsolve/ldl_lsolve.h"
#include "../ldl_dsolve/ldl_dsolve.h"
#include "../ldl_ltsolve/ldl_ltsolve.h"
#include "xlr8_definitions.h"

#include <hls_stream.h>


void xlr8_read_ir(spm_info info, int ir[], hls::stream<int>& ir_stream) {
    READ_IR: for (int i = 0; i < info.nnz; i += INT_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_NNZ/INT_BURST) max=(MAX_NNZ/INT_BURST)
        int chunk_size = INT_BURST;
        if ((i + chunk_size) > info.nnz) chunk_size = info.nnz - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=INT_BURST
            ir_stream.write(ir[i + j]);
        }
    }
}

 void xlr8_read_pr(spm_info info, DTYPE pr[], hls::stream<DTYPE>& pr_stream) {
    READ_PR: for (int i = 0; i < info.nnz; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_NNZ/DTYPE_BURST) max=(MAX_NNZ/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.nnz) chunk_size = info.nnz - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            pr_stream.write(pr[i + j]);
        }
    }
}

void xlr8_read_jc(spm_info info, int jc[], hls::stream<int>& jc_stream) {
    int jc_size = info.n + 1;
    READ_JC: for (int i = 0; i < jc_size; i += INT_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_JC/INT_BURST) max=(MAX_JC/INT_BURST)
        int chunk_size = INT_BURST;
        if ((i + chunk_size) > jc_size) chunk_size = jc_size - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=INT_BURST
            jc_stream.write(jc[i + j]);            
        }
    }
}

void xlr8_read_in(int len, DTYPE in_vec[], hls::stream<DTYPE>& in_vec_stream) {
    READ_X_IN: for (int i = 0; i < len; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_IN_VEC/DTYPE_BURST) max=(MAX_IN_VEC/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > len) chunk_size = len - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            in_vec_stream.write(in_vec[i + j]);
        }   
    }
}

void xlr8_read_ir_inv(spm_info info, int ir[], hls::stream<int>& ir_stream) {
    int nnz = info.nnz;
    int buff[INT_BURST] = {0};
    READ_IR_IN: for (int i = nnz-1; i >= 0; i--) {
        ir_stream.write(ir[i]);
    }
}

 void xlr8_read_pr_inv(spm_info info, DTYPE pr[], hls::stream<DTYPE>& pr_stream) {
    int nnz = info.nnz;
    READ_PR_IN: for (int i = nnz - 1; i >= 0; i--) {
        pr_stream.write(pr[i]);
    }
}

void xlr8_read_jc_inv(spm_info info, int jc[], hls::stream<int>& jc_stream) {
    int jc_size = info.n + 1;
    READ_JC_IN: for (int i = jc_size - 1; i >= 0; i--) {
        jc_stream.write(jc[i]);
    }
}

void xlr8_read_in_inv(int len, DTYPE in_vec[], hls::stream<DTYPE>& in_vec_stream) {
    READ_X_IN: for (int i = len-1; i >= 0; i--) {
        in_vec_stream.write(in_vec[i]);        
    }
}

void xlr8_ldl_writeback(spm_info info, hls::stream<DTYPE>& X_stream, DTYPE X[]) {
    for (int i = 0; i < info.n; i++) {
        #pragma HLS PIPELINE
        DTYPE value = X_stream.read();
        X[i] = value;
    }
}

void xlr8_spmv(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag a, flag new_vec, DTYPE* gbuff) {
    #pragma HLS DATAFLOW
    
    hls::stream<int> g_jc_stream, g_ir_stream;
    hls::stream<DTYPE> g_pr_stream, g_in_vec_stream, g_out_vec_stream;
    #pragma HLS STREAM variable=g_jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=g_ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_in_vec_stream depth=MAX_IN_VEC
    #pragma HLS STREAM variable=g_out_vec_stream depth=MAX_OUT_VEC

    xlr8_read_ir(info, ir, g_ir_stream);
    xlr8_read_jc(info, jc, g_jc_stream);
    xlr8_read_pr(info, pr, g_pr_stream);
    xlr8_read_in(info.n, x, g_in_vec_stream);
    spmv_compute_vector(info, g_jc_stream, g_ir_stream, g_pr_stream, g_in_vec_stream, g_out_vec_stream, gbuff);
    spmv_writeback(info, g_out_vec_stream, y, a, new_vec);
}

void xlr8_spmtvm(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag new_vec, flag skip_diagonal, DTYPE* gbuff) {
    #pragma HLS DATAFLOW

    hls::stream<int> g_jc_stream, g_ir_stream;
    hls::stream<DTYPE> g_pr_stream, g_in_vec_stream, g_out_vec_stream;
    #pragma HLS STREAM variable=g_jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=g_ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_in_vec_stream depth=MAX_IN_VEC
    #pragma HLS STREAM variable=g_out_vec_stream depth=MAX_OUT_VEC

    xlr8_read_ir(info, ir, g_ir_stream);
    xlr8_read_jc(info, jc, g_jc_stream);
    xlr8_read_pr(info, pr, g_pr_stream);
    xlr8_read_in(info.m, x, g_in_vec_stream);
    spmtv_compute(info, g_jc_stream, g_ir_stream, g_pr_stream, g_in_vec_stream, g_out_vec_stream, skip_diagonal, gbuff);
    spmtv_writeback(info, g_out_vec_stream, y, new_vec);
}

void xlr8_lsolve(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE B[], DTYPE X[], DTYPE* gbuff) {
    #pragma HLS DATAFLOW

    hls::stream<int> g_jc_stream, g_ir_stream;
    hls::stream<DTYPE> g_pr_stream, g_in_vec_stream, g_out_vec_stream;
    #pragma HLS STREAM variable=g_jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=g_ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_in_vec_stream depth=MAX_IN_VEC
    #pragma HLS STREAM variable=g_out_vec_stream depth=MAX_OUT_VEC

    xlr8_read_ir(info, ir, g_ir_stream);
    xlr8_read_jc(info, jc, g_jc_stream);
    xlr8_read_pr(info, pr, g_pr_stream);
    xlr8_read_in(info.n, B, g_in_vec_stream);
    lsolve(info, g_jc_stream, g_ir_stream, g_pr_stream, g_in_vec_stream, g_out_vec_stream, gbuff);
    xlr8_ldl_writeback(info, g_out_vec_stream, X);
}


void xlr8_dsolve(spm_info info, DTYPE pr[], DTYPE X[], DTYPE B[]) {
    #pragma HLS DATAFLOW

    hls::stream<int> g_jc_stream, g_ir_stream;
    hls::stream<DTYPE> g_pr_stream, g_in_vec_stream, g_out_vec_stream;
    #pragma HLS STREAM variable=g_jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=g_ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_in_vec_stream depth=MAX_IN_VEC
    #pragma HLS STREAM variable=g_out_vec_stream depth=MAX_OUT_VEC

    xlr8_read_pr(info, pr, g_pr_stream);
    xlr8_read_in(info.n, B, g_in_vec_stream);
    dsolve(info, g_pr_stream, g_in_vec_stream, g_out_vec_stream);
    xlr8_ldl_writeback(info, g_out_vec_stream, X);
}

void xlr8_ltsolve(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE B[], DTYPE X[], DTYPE* gbuff) {
    #pragma HLS DATAFLOW

    hls::stream<int> g_jc_stream, g_ir_stream;
    hls::stream<DTYPE> g_pr_stream, g_in_vec_stream, g_out_vec_stream;
    #pragma HLS STREAM variable=g_jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=g_ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=g_in_vec_stream depth=MAX_IN_VEC
    #pragma HLS STREAM variable=g_out_vec_stream depth=MAX_OUT_VEC

    xlr8_read_ir_inv(info, ir, g_ir_stream);
    xlr8_read_jc_inv(info, jc, g_jc_stream);
    xlr8_read_pr_inv(info, pr, g_pr_stream);
    xlr8_read_in_inv(info.n, B, g_in_vec_stream);
    ltsolve(info, g_jc_stream, g_ir_stream, g_pr_stream, g_in_vec_stream, g_out_vec_stream, gbuff);
    xlr8_ldl_writeback(info, g_out_vec_stream, X);
}

void xlr8(fn_code function, spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE in_vec[], DTYPE out_vec[], params p) {    
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=MAX_JC port=jc offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=MAX_NNZ port=ir offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=MAX_NNZ port=pr offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=x_maxi depth=MAX_IN_VEC port=in_vec offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=y_maxi depth=MAX_OUT_VEC port=out_vec offset=slave max_read_burst_length=DTYPE_BURST max_write_burst_length=DTYPE_BURST max_widen_bitwidth=1024

    #pragma HLS INTERFACE s_axilite port=function bundle=control
    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=in_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=out_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=p bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
    #pragma HLS BIND_STORAGE variable=gbuff type=ram_t2p impl=BRAM

    flag f1 = p & 0x1;
    flag f2 = (p >> 1) & 0x1;
    switch (function) {
        case FN_SPMV:
            xlr8_spmv(info, jc, ir, pr, in_vec, out_vec, f1, f2, gbuff);
            break;
        case FN_SPMTVM:
            xlr8_spmtvm(info, jc, ir, pr, in_vec, out_vec, f1, f2, gbuff);
            break;
        case FN_LSOLVE:
            xlr8_lsolve(info, jc, ir, pr, in_vec, out_vec, gbuff);
            break;
        case FN_DSOLVE:
            xlr8_dsolve(info, pr, out_vec, in_vec);
            break;
        case FN_LTSOLVE:
            //xlr8_ltsolve(info, jc, ir, pr, in_vec, out_vec, gbuff);
            break;
        default:
            break;
    }    
}
