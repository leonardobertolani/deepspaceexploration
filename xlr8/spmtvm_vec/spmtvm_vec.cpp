#include "spmtvm_vec.h"
#include "hls_vector.h"

void spmtvm_vec_compute(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& x, hls::stream<double_vec>& y, flag skip_diagonal, DTYPE* x_buff) {
    int n = info.n;
    int x_size = info.m >> 2;
    if (info.m % 4) {
        x_size++;
    } 
    COPY_X_BUFF: for (int i = 0; i < x_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=30
        double_vec t = x.read();
        for (int j = 0; j < 4; j++) {
            x_buff[i*4 + j] = t[j];
        }
    }
    
    int cols_to_read = n >> 2;
    if (n % 4 != 0) {
        cols_to_read++;
    }
    SPMTVMVEC_COMP_OUT: for (int i = 0; i < cols_to_read; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=100 max=223
        double_vec y_t = 0.0;
        col_len cls[4];
        READ_CLS: for (int j = 0; j < 4; j++) {
            cls[j] = cols_len.read();            
        }

        SPMTVMVEC_COMP_COL: for (int j = 0; j < 4; j++) {
            int col_idx = i*4 + j;
            col_len cl = cls[j];
            int vec_to_read = cl.cl_padded >> 2;

            double t = 0.0;
            if (cl.cl >= 4) {            
                SPMTVMVEC_COMP_MUL: for (int k = 0; k < vec_to_read; k++) {
                    #pragma HLS PIPELINE II=4
                    int4_vec ir_vec = ir_padded.read();
                    double_vec pr_vec = pr_padded.read();
                    double_vec x_vec;
                    SPMTVMVEC_COMP_XVEC: for (int z = 0; z < 4; z++) {
                        #pragma HLS UNROLL
                        int iri = ir_vec[z];
                        if (!skip_diagonal.is_zero() && col_idx == iri) {
                            x_vec[z] = 0.0;
                        } else {                        
                            x_vec[z] = x_buff[iri];
                        }
                    }

                    double t_inc = 0.0;
                    double_vec t_res = x_vec * pr_vec;
                    double t_res_1 = t_res[0] + t_res[1];
                    double t_res_2 = t_res[2] + t_res[3];
                    t_inc = t_res_1 + t_res_2;
                    t += t_inc;
                }
            } else if (cl.cl > 0) {
                int4_vec ir_vec = ir_padded.read();
                double_vec pr_vec = pr_padded.read();
                SPMTVEC_COMP_SMALL: for (int k = 0; k < cl.cl; k++) {
                    #pragma HLS PIPELINE off
                    int iri = ir_vec[k];
                    if (!(!skip_diagonal.is_zero() && col_idx == iri)) {
                        double x_v = x_buff[iri];
                        t += x_v * pr_vec[k];
                    }                    
                }
            }
            
            y_t[j] = t;
        }

        y.write(y_t);
    }
}

void spmtvm_vec_xsize(spm_info info, int& x_size) {
    int t = info.m >> 2;
    if (info.m % 4 != 0) {
        t++;
    }
    x_size = t;
}

void spmtvm_vec_writeback_nv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int vec_to_write = info.n >> 2;
    if (info.n % 4 != 0) {
        vec_to_write++;
    }
    WB_NA_NV: for (int i = 0; i < vec_to_write; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=30
        double_vec v = y_stream.read();
        for (int k = 0; k < 4; k++) {
            v[k] = -v[k];
        }
        y[i] = v;
    }
}

void spmtvm_vec_writeback_nnv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int vec_to_write = info.n >> 2;
    if (info.n % 4 != 0) {
        vec_to_write++;
    }
    WB_NA_NNV: for (int i = 0; i < vec_to_write; i++) {
        #pragma HLS PIPELINE II=20
        #pragma HLS LOOP_TRIPCOUNT min=10 max=30
        double_vec t = y[i];
        double_vec v = y_stream.read();
        t -= v;
        y[i] = t;
    }
}

void spmtvm_vec_writeback(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[], flag new_vec) {
    if (new_vec.is_zero()) {
        spmtvm_vec_writeback_nnv(info, y_stream, y);
    } else {
        spmtvm_vec_writeback_nv(info, y_stream, y);
    }
}

void spmtvm_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec x[], double_vec y[], flag new_vec, flag skip_diagonal) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=28 port=jc offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=80 port=ir offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=160 port=pr offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=x_maxi depth=55 port=x offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=y_maxi depth=56 port=y offset=slave max_widen_bitwidth=256

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

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
    #pragma HLS BIND_STORAGE variable=gbuff type=ram_t2p impl=BRAM
    
    hls::stream<int_vec> jc_stream("jc_stream"), ir_stream("ir_stream");
    hls::stream<double_vec> pr_stream("pr_stream"), x_stream("x_stream"), y_stream("y_stream"); 
    hls::stream<double_vec> pr_padded_stream("pr_padded_stream");
    hls::stream<col_len> cols_lens_stream("cols_lens_stream");
    hls::stream<int4_vec> ir_padded_stream("ir_padded_stream");

    #pragma HLS STREAM variable=jc_stream depth=28
    #pragma HLS STREAM variable=ir_stream depth=75
    #pragma HLS STREAM variable=pr_stream depth=130
    #pragma HLS STREAM variable=pr_padded_stream depth=110
    #pragma HLS STREAM variable=x_stream  depth=55
    #pragma HLS STREAM variable=y_stream  depth=56
    #pragma HLS STREAM variable=cols_lens_stream depth=90
    #pragma HLS STREAM variable=ir_padded_stream depth=110

    int x_size;
    spmtvm_vec_xsize(info, x_size);
    xlr8_read_jc_vec(info, jc, jc_stream);
    xlr8_read_ir_vec(info, ir, ir_stream);
    xlr8_read_pr_vec(info, pr, pr_stream);
    xlr8_read_in_vec(x_size, x, x_stream);
    xlr8_vec_load_padded(info, jc_stream, ir_stream, pr_stream, cols_lens_stream, pr_padded_stream, ir_padded_stream);
    spmtvm_vec_compute(info, cols_lens_stream, ir_padded_stream, pr_padded_stream, x_stream, y_stream, skip_diagonal, gbuff);
    spmtvm_vec_writeback(info, y_stream, y, new_vec);
}
