#include "sparse_mtvm.h"
#include <hls_stream.h>

void spmtv_read_ir(spm_info info, int ir[], hls::stream<int>& ir_stream) {
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

void spmtv_read_pr(spm_info info, DTYPE pr[], hls::stream<DTYPE>& pr_stream) {
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

void spmtv_read_jc(spm_info info, int jc[], hls::stream<int>& jc_stream) {
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

void spmtv_read_x(spm_info info, DTYPE x[], hls::stream<DTYPE>& x_stream) {
    READ_X: for (int i = 0; i < info.m; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_M/DTYPE_BURST) max=(MAX_M/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.m) chunk_size = info.m - i;
        for (int j = 0; j < chunk_size; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=DTYPE_BURST
            x_stream.write(x[i + j]);
        }   
    }
}

void add4(DTYPE v[], DTYPE &out) {
    DTYPE v0 = v[0] + v[1];
    DTYPE v1 = v[2] + v[3];
    DTYPE v2 = v0 + v1;
    out = v2;    
}

void add8(DTYPE v[], DTYPE& out) {
    DTYPE v0, v1;
    add4(v, v0);
    add4(&v[4], v1);
    DTYPE v2 = v0 + v1;
    out = v2;
}

void add16(DTYPE v[], DTYPE& out) {
    DTYPE v0, v1;
    add8(v, v0);
    add8(&v[8], v1);
    DTYPE v2 = v0 + v1;
    out = v2;
}

// singular vars 7477
// buff 4 9709 - 7477
// buff 8 8593
// buff 16 9709
#define ADDER_DEPTH 4
void spmtv_compute_diag(spm_info info, DTYPE x[], hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& y) {
    int n = info.n;
    
    int i0 = jc.read();
    int i1 = jc.read();
    D_OUT_LOOP: for (int j = 0; j < n; j++) {
        #pragma HLS PIPELINE II=4
        DTYPE y_t = 0.0;

        DTYPE y_buff[ADDER_DEPTH] = {0.0};
        #pragma HLS ARRAY_PARTITION variable=y_buff dim=1 type=complete
        D_IN_LOOP: for (int i = i0; i < i1; i++) {
            #pragma HLS PIPELINE II=4
            DTYPE m_val = pr.read();
            int iri = ir.read();
            int iy = i % ADDER_DEPTH;
            y_buff[iy] += m_val * x[iri];
        }
        DTYPE t;
        add4(y_buff, t);
        y_t = t;                
        y.write(y_t);

        if (j < info.n - 1) {
            i0 = i1;
            i1 = jc.read();
        }
    }
}

// baseline w/ diag 
// no continue 8569
// continue 8569
// zero 8569
// with add4 7477
void spmtv_compute_nodiag(spm_info info, DTYPE x[], hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& y) {
    int n = info.n;
    
    int i0 = jc.read();
    int i1 = jc.read();
    ND_OUT_LOOP: for (int j = 0; j < n; j++) {
        #pragma HLS PIPELINE II=4
        DTYPE y_t = 0.0;

        DTYPE y_buff[ADDER_DEPTH] = {0.0};
        #pragma HLS ARRAY_PARTITION variable=y_buff dim=1 type=complete
        ND_IN_LOOP: for (int i = i0; i < i1; i++) {
            #pragma HLS PIPELINE II=4
            DTYPE m_val = pr.read();
            int iri = ir.read();
            if (j == iri) {
                continue;                
            }            
            int iy = i % ADDER_DEPTH;
            y_buff[iy] += m_val * x[iri];
        }
        DTYPE t;
        add4(y_buff, t);
        y_t = t;                
        y.write(y_t);

        if (j < info.n - 1) {
            i0 = i1;
            i1 = jc.read();
        }
    }
}

void spmtv_compute(spm_info info, hls::stream<int>& jc, hls::stream<int>& ir, hls::stream<DTYPE>& pr, hls::stream<DTYPE>& x, hls::stream<DTYPE>& y, flag skip_diagonal, DTYPE *x_buff) {
    int x_size = info.m;
    COPY_X_BUFF: for (int i = 0; i < x_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MIN_M max=MAX_M
        DTYPE t = x.read();
        x_buff[i] = t;
    }

    if (skip_diagonal.is_zero()) {
        spmtv_compute_diag(info, x_buff, jc, ir, pr, y);
    } else {
        spmtv_compute_nodiag(info, x_buff, jc, ir, pr, y);
    }
}

void spmtv_writeback_nv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    WB_NA_NV: for (int i = 0; i < info.n; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MIN_N max=MAX_N
        DTYPE v = y_stream.read();
        y[i] = -v;
    }
}

void spmtv_writeback_nnv(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[]) {
    DTYPE y_buff[DTYPE_BURST] = { 0.0 };
    WB_NA_NNV: for (int i = 0; i < info.n; i += DTYPE_BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=(MIN_N/DTYPE_BURST) max=(MAX_N/DTYPE_BURST)
        int chunk_size = DTYPE_BURST;
        if ((i + chunk_size) > info.n) chunk_size = info.n - i;
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

void spmtv_writeback(spm_info info, hls::stream<DTYPE>& y_stream, DTYPE y[], flag new_vec) {
    if (new_vec.is_zero()) {
        spmtv_writeback_nnv(info, y_stream, y);
    } else {
        spmtv_writeback_nv(info, y_stream, y);
    }
}

void sparse_mtvm(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE x[], DTYPE y[], flag new_vec, flag skip_diagonal) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=MAX_JC port=jc offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=MAX_NNZ port=ir offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=MAX_NNZ port=pr offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=x_maxi depth=MAX_M port=x offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=1024
    #pragma HLS INTERFACE mode=m_axi bundle=y_maxi depth=MAX_N port=y offset=slave max_read_burst_length=DTYPE_BURST max_write_burst_length=DTYPE_BURST max_widen_bitwidth=1024

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=new_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=skip_diagonal bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    hls::stream<int> jc_stream, ir_stream;
    hls::stream<DTYPE> pr_stream, x_stream, y_stream;

    #pragma HLS STREAM variable=jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=x_stream  depth=MAX_M
    #pragma HLS STREAM variable=y_stream  depth=MAX_N

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
    #pragma HLS BIND_STORAGE variable=gbuff type=ram_t2p impl=BRAM

    spmtv_read_ir(info, ir, ir_stream);
    spmtv_read_jc(info, jc, jc_stream);
    spmtv_read_pr(info, pr, pr_stream);
    spmtv_read_x(info, x, x_stream);
    spmtv_compute(info, jc_stream, ir_stream, pr_stream, x_stream, y_stream, skip_diagonal, gbuff);
    spmtv_writeback(info, y_stream, y, new_vec);
}
