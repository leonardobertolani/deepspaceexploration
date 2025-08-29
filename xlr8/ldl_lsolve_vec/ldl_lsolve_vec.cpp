#include <stdio.h>
#include <hls_vector.h>
#include "hls_stream.h"
#include "ldl_lsolve_vec.h"

void lsolve_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& X, DTYPE* result) {    
    int n_cols = info.n;
    int cols_to_read = n_cols >> 2;
    if (n_cols % 4 != 0) {
        cols_to_read++;
    }

    EXTERNAL_LOOP_LSOLVE: for (int i = 0; i < cols_to_read; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143

        CL_LOOP_LSOLVE: for (int z = 0; z < 4; z++) {
            col_len cl;
            cl = cols_len.read();
            int vec_to_read = cl.cl_padded >> 2;

            if (cl.cl >= 4) {
                double_vec x_vec;
                int xi = (i << 2) + z;
                double xv = result[xi];
                x_vec[0] = xv;
                x_vec[1] = xv;
                x_vec[2] = xv;
                x_vec[3] = xv;                
                COMPUTE_COL_LSOLVE: for (int j = 0; j < vec_to_read; j++) {
                    #pragma HLS PIPELINE II=4
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=10 avg=1
                    #pragma HLS DEPENDENCE variable=result type=inter false

                    double_vec m_vec = pr_padded.read();
                    double_vec t = m_vec * x_vec;

                    int4_vec ir_vec = ir_padded.read();
                    double temp[4];
                    #pragma HLS ARRAY_PARTITION variable=temp dim=1 type=complete

                    READ_B_LOOP_LSOLVE: for (int k = 0; k < 4; k++) {
                        temp[k] = result[ir_vec[k]];
                    }

                    SUB_LOOP_LSOLVE: for (int k = 0; k < 4; k++) {
                        temp[k] -= t[k];
                    }

                    WRITE_RES_LOOP_LSOLVE: for (int k = 0; k < 4; k++) {
                        result[ir_vec[k]] = temp[k];
                    }
                }
            } else if (cl.cl > 0){
                int xi = i << 2;                
                double x_val = result[xi + z];
                double_vec m_vec = pr_padded.read();
                int4_vec ir_vec = ir_padded.read();

                COMPUTE_SINGLE_LSOLVE: for (int j = 0; j < cl.cl; j++) {
                    #pragma HLS PIPELINE 
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=3 avg=1           
                    #pragma HLS DEPENDENCE variable=result type=inter false
                    int iri = ir_vec[j];
                    double t = x_val * m_vec[j];
                    double temp_sub = result[iri];
                    temp_sub -= t;
                    result[iri] = temp_sub;                    
                }
            }
        }
    }
    
    int X_size = info.m >> 2;
    if (info.m % 4 > 0) {
        X_size++;
    }    
    double_vec X_vec;
    WRITE_X_LSOLVE: for (int i = 0; i < X_size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143
        for (int j = 0; j < 4; j++) {
            X_vec[j] = result[i*4 + j];
        }
        X.write(X_vec);
    }
}

void lsolve_vec_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]) {
    int X_size = info.m >> 2;
    if (info.m % 4 > 0) {
        X_size++;
    }    
    FINAL_WB_LSOLVE: for (int i = 0; i < X_size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143
        double_vec v = X_stream.read();
        X[i] = v;
    }
}


void lsolve_dvec_to_buff(double_vec vec[], DTYPE* buff, size_t len) {
    int chunk_number = len >> 2;

    if(len%4 != 0)
        chunk_number++;

    for (size_t i = 0; i < chunk_number; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143
        buff[i*4 + 0] = vec[i][0];
        buff[i*4 + 1] = vec[i][1];
        buff[i*4 + 2] = vec[i][2];
        buff[i*4 + 3] = vec[i][3];
    }
}


void ldl_lsolve_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=72 port=jc offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=122 port=ir offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=243 port=pr offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=B_maxi depth=143 port=B offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=X_maxi depth=143 port=X offset=slave max_widen_bitwidth=256

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=X bundle=control 
    #pragma HLS INTERFACE s_axilite port=return bundle=control


    #pragma HLS DATAFLOW

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
    lsolve_dvec_to_buff(B, gbuff, info.n);
    #pragma HLS BIND_STORAGE variable=gbuff type=ram_t2p impl=BRAM

    
    hls::stream<int_vec> jc_stream("jc_stream"), ir_stream("ir_stream");
    hls::stream<double_vec> pr_stream("pr_stream"), X_stream("X_stream"); 
    hls::stream<double_vec> pr_padded_stream("pr_padded_stream");
    hls::stream<col_len> cols_lens_stream("cols_lens_stream");
    hls::stream<int4_vec> ir_padded_stream("ir_padded_stream");

    #pragma HLS STREAM variable=jc_stream depth=72
    #pragma HLS STREAM variable=ir_stream depth=122
    #pragma HLS STREAM variable=pr_stream depth=243
    #pragma HLS STREAM variable=pr_padded_stream depth=350
    #pragma HLS STREAM variable=X_stream  depth=120
    #pragma HLS STREAM variable=cols_lens_stream depth=350
    #pragma HLS STREAM variable=ir_padded_stream depth=350

    xlr8_read_jc_vec(info, jc, jc_stream);
    xlr8_read_ir_vec(info, ir, ir_stream);
    xlr8_read_pr_vec(info, pr, pr_stream);
    xlr8_vec_load_padded(info, jc_stream, ir_stream, pr_stream, cols_lens_stream, pr_padded_stream, ir_padded_stream);
    lsolve_vec_compute_pad(info, cols_lens_stream, ir_padded_stream, pr_padded_stream, X_stream, gbuff);
    lsolve_vec_writeback(info, X_stream, X);
}
