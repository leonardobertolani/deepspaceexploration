#include "sparse_mv.h"
#include <hls_vector.h>
#include <hls_stream.h>

void spmv_read_ir(spm_info info, int ir[], hls::stream<int>& ir_stream) {
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

void spmv_read_pr(spm_info info, DTYPE pr[], hls::stream<DTYPE>& pr_stream) {
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

void spmv_read_jc(spm_info info, int jc[], hls::stream<int>& jc_stream) {
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

void spmv_read_x(spm_info info, DTYPE x[], hls::stream<DTYPE>& x_stream) {
    READ_X: for (int i = 0; i < info.n; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_N/DTYPE_BURST) max=(MAX_N/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.n) chunk_size = info.n - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            x_stream.write(x[i + j]);
        }   
    }
}

// 5479
void spmv_compute(spm_info info, hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& x, hls::stream<DTYPE>& y, DTYPE* result) {
    int result_size = info.m;

    int i0 = jc.read();
    int i1 = jc.read();
    OUT_COMPUTE_SPMV: for (int j = 0; j < info.n; j++) {
        #pragma HLS PIPELINE
        DTYPE x_val = x.read();
        IN_COMPUTE_SPMV: for (int i = i0; i < i1; i++) {
            #pragma HLS PIPELINE II=4
            int iri = ir.read();
            DTYPE m_val = pr.read();
            result[iri] += x_val * m_val;
        }

        if (j < info.n - 1) {
            i0 = i1;
            i1 = jc.read();
        }
    }

    WRITE_RESULT: for (int i = 0; i < result_size; i++) {
        #pragma HLS PIPELINE
        y.write(result[i]);
    }
}

// 5257 8vec
// 5257 16vec
// 5257 4vec
#define VECTOR_SIZE 4
void spmv_compute_vector(spm_info info, hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& x, hls::stream<DTYPE>& y, DTYPE* result) {
    int result_size = info.m;

    hls::vector<DTYPE, VECTOR_SIZE> x_vec;
    hls::vector<DTYPE, VECTOR_SIZE> m_vec;
    int ir_vec[VECTOR_SIZE];
    int idx_vec = 0;

    int i0 = jc.read();
    int i1 = jc.read();

    OUT_COMPUTE_SPMV: for (int j = 0; j < info.n; j++) {
        #pragma HLS PIPELINE
        DTYPE x_val = x.read();

        int col_len = i1 - i0;
        if (col_len >= VECTOR_SIZE) {
                COMPUTE_SPMV_VEC: for (int i = i0; i < i1; i++) {
                    #pragma HLS PIPELINE II=4
                    if (idx_vec >= VECTOR_SIZE) {
                        x_vec *= m_vec;
                        ADD_RESULT: for (int k = 0; k < VECTOR_SIZE; k++) {
                            #pragma HLS UNROLL
                            int r_idx = ir_vec[k];
                            result[r_idx] += x_vec[k];
                        }
                        idx_vec = 0;
                    }            
                    int iri = ir.read();
                    DTYPE m_val = pr.read();
                    x_vec[idx_vec] = x_val;
                    m_vec[idx_vec] = m_val;
                    ir_vec[idx_vec] = iri;
                    idx_vec++;   

                if (idx_vec > 0) {
                    x_vec *= m_vec;
                    for (int k = 0; k < idx_vec; k++) {
                        #pragma HLS LOOP_FLATTEN off
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=VECTOR_SIZE
                        int r_idx = ir_vec[k];
                        result[r_idx] += x_vec[k];
                    }
                    idx_vec = 0;
                }
            }
        } else {   
            COMPUTE_SPMV_SINGULAR: for (int i = i0; i < i1; i++) {
                #pragma HLS PIPELINE II=4
                int iri = ir.read();
                DTYPE m_val = pr.read();
                result[iri] += x_val * m_val;
            }
        }


        if (j < info.n - 1) {
            i0 = i1;
            i1 = jc.read();
        }
    }  

    WRITE_RESULT: for (int i = 0; i < result_size; i++) {
        #pragma HLS PIPELINE
        y.write(result[i]);
    }
}

void spmv_writeback_a_nv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    WB_A_NV: for (int i = 0; i < info.m; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MIN_M max=MAX_M        
        DTYPE v = y_stream.read();
        y[i] = v;
    }    
}

void spmv_writeback_na_nv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    WB_NA_NV: for (int i = 0; i < info.m; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MIN_M max=MAX_M
        DTYPE v = y_stream.read();
        y[i] = -v;
    }
}

void spmv_writeback_a_nnv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    DTYPE y_buff[DTYPE_BURST] = { 0.0 };
    WB_A_NNV: for (int i = 0; i < info.m; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_M/DTYPE_BURST) max=(MAX_M/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.m) chunk_size = info.m - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST            
            y_buff[j] = y[i+j];
        }

        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            DTYPE v = y_stream.read();
            y_buff[j] += v;
            y[i+j] = y_buff[j];
        }
    }    
}

void spmv_writeback_na_nnv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    DTYPE y_buff[DTYPE_BURST] = { 0.0 };
    WB_NA_NNV: for (int i = 0; i < info.m; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_M/DTYPE_BURST) max=(MAX_M/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.m) chunk_size = info.m - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST            
            y_buff[j] = y[i+j];
        }

        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            DTYPE v = y_stream.read();
            y_buff[j] -= v;
            y[i+j] = y_buff[j];
        }
    } 
}

void spmv_writeback(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[], flag a, flag new_vec) {
    if (!a.is_zero() && !new_vec.is_zero()) {
        spmv_writeback_a_nv(info ,y_stream, y);
    } else if (a.is_zero() && !new_vec.is_zero()) {
        spmv_writeback_na_nv(info ,y_stream, y);
    } else if (!a.is_zero() && new_vec.is_zero()) {
        spmv_writeback_a_nnv(info ,y_stream, y);
    } else {
        spmv_writeback_na_nnv(info ,y_stream, y);
    }
}

void sparse_mv(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag a, flag new_vec) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=MAX_JC port=jc offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=MAX_NNZ port=ir offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=MAX_NNZ port=pr offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=x_maxi depth=MAX_N port=x offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=y_maxi depth=MAX_M port=y offset=slave max_read_burst_length=DTYPE_BURST max_write_burst_length=DTYPE_BURST max_widen_bitwidth=1024

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=a bundle=control
    #pragma HLS INTERFACE s_axilite port=new_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    hls::stream<int> jc_stream, ir_stream;
    hls::stream<DTYPE> pr_stream, x_stream, y_stream;

    #pragma HLS STREAM variable=jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=x_stream  depth=MAX_N 
    #pragma HLS STREAM variable=y_stream  depth=MAX_M

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
    #pragma HLS BIND_STORAGE variable=gbuff type=ram_t2p impl=BRAM

    spmv_read_ir(info, ir, ir_stream);
    spmv_read_jc(info, jc, jc_stream);
    spmv_read_pr(info, pr, pr_stream);
    spmv_read_x(info, x, x_stream);    
    spmv_compute_vector(info, jc_stream, ir_stream, pr_stream, x_stream, y_stream, gbuff);
    spmv_writeback(info, y_stream, y, a, new_vec);
}
