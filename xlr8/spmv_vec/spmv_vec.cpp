#include "spmv_vec.h"
#include "hls_stream.h"
#include "hls_vector.h"

void spmv_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& x, hls::stream<double_vec>& y, DTYPE* result) {
    int n_cols = info.n;
    int vec_to_read = n_cols >> 2;
    if (n_cols % 4 != 0) {
        vec_to_read++;
    }

    SPMVEC_OUT: for (int i = 0; i < vec_to_read; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=60

        double_vec curr_x = x.read();
        SPMVEC_COLUMNS: for (int z = 0; z < 4; z++) {
            int el_read = (i >> 2) + z;
            col_len cl;    
            cl = cols_len.read();
            int vec_to_read = cl.cl_padded >> 2;

            if (cl.cl >= 4) {
                double_vec x_vec;
                double xv = curr_x[z];
                x_vec[0] = xv;
                x_vec[1] = xv;
                x_vec[2] = xv;
                x_vec[3] = xv;                
                
                SPMVEC_COMP_COL: for (int j = 0; j < vec_to_read; j++) {
                    #pragma HLS PIPELINE II=4
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=28
                    #pragma HLS DEPENDENCE variable=result type=inter false

                    double_vec m_vec = pr_padded.read();
                    double_vec t = m_vec * x_vec;
                    int4_vec ir_vec = ir_padded.read();

                    double_vec res_vec;
                    SPMVEC_READ_RES: for (int k = 0; k < 4; k++) {
                        #pragma HLS UNROLL
                        int iri = ir_vec[k];
                        res_vec[k] = result[iri];
                    }

                    res_vec += t;

                    SPMVEC_WRITE_RES: for (int k = 0; k < 4; k++) {
                        #pragma HLS UNROLL
                        int iri = ir_vec[k];
                        result[iri] = res_vec[k];                                                
                    }
                }
            } else if (cl.cl > 0){
                double x_val = curr_x[z];
                double_vec m_vec = pr_padded.read();
                int4_vec ir_vec = ir_padded.read();
                SPMVEC_COMP_SINGLE: for (int j = 0; j < cl.cl; j++) {
                    #pragma HLS PIPELINE off
                    #pragma HLS DEPENDENCE variable=result type=inter false
                    int iri = ir_vec[j];
                    double t = x_val * m_vec[j];
                    result[iri] += t;
                }
            }
        }
    }
    
    int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    double_vec y_vec;
    SPMVEC_WRITE_Y: for (int i = 0; i < y_size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        for (int j = 0; j < 4; j++) {
            y_vec[j] = result[i*4 + j];
        }
        y.write(y_vec);
    }
}

void spmv_vec_writeback_a_nv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    WB_ANV: for (int i = 0; i < y_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        #pragma HLS PIPELINE
        double_vec v = y_stream.read();
        y[i] = v;
    }
}

void spmv_vec_writeback_na_nv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    WB_NANV: for (int i = 0; i < y_size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        double_vec v = y_stream.read();
        for (int j = 0; j < 4; j++) {
            v[j] = -v[j];
        }
        y[i] = v;
    }
}

void spmv_vec_writeback_a_nnv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    WB_ANNV: for (int i = 0; i < y_size; i++) {
        #pragma HLS PIPELINE II=20
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        double_vec t = y[i];
        double_vec v = y_stream.read();
        t += v;
        y[i] = t;
    }
}

void spmv_vec_writeback_na_nnv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[]) {
    int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    WB_NANNV: for (int i = 0; i < y_size; i++) {
        #pragma HLS PIPELINE II=20
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        double_vec t = y[i];
        double_vec v = y_stream.read();
        t -= v;
        y[i] = t;
    }
}

void spmv_vec_writeback_nnv(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[], flag a) {
        int y_size = info.m >> 2;
    if (info.m % 4 > 0) {
        y_size++;
    }    
    WB_NANNV: for (int i = 0; i < y_size; i++) {
        #pragma HLS PIPELINE II=20
        #pragma HLS LOOP_TRIPCOUNT min=28 max=56
        double_vec t = y[i];
        double_vec v = y_stream.read();
        if (a.is_zero()) {
            t -= v;
        } else {
            t += v;
        }
        y[i] = t;
    }
}

void spmv_vec_writeback(spm_info info, hls::stream<double_vec>& y_stream, double_vec y[], flag a, flag new_vec) {
    if (!a.is_zero() && !new_vec.is_zero()) {
        spmv_vec_writeback_a_nv(info, y_stream, y);
    } else if (a.is_zero() && !new_vec.is_zero()) {
        spmv_vec_writeback_na_nv(info, y_stream, y);
    } else {
        spmv_vec_writeback_nnv(info, y_stream, y, a);        
    }    
}

void spmv_vec_xsize(spm_info info, int& x_size) {
    int xs = info.n >> 2;
    if (info.n % 4 > 0) {
        xs++;
    }
    x_size = xs;
}

void spmv_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec x[], double_vec y[], flag a, flag new_vec) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=28 port=jc offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=80 port=ir offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=160 port=pr offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=x_maxi depth=56 port=x offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=y_maxi depth=55 port=y offset=slave max_widen_bitwidth=256

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
    #pragma HLS STREAM variable=x_stream  depth=56
    #pragma HLS STREAM variable=y_stream  depth=55
    #pragma HLS STREAM variable=cols_lens_stream depth=60
    #pragma HLS STREAM variable=ir_padded_stream depth=110

    int x_size;
    spmv_vec_xsize(info, x_size);
    xlr8_read_jc_vec(info, jc, jc_stream);
    xlr8_read_ir_vec(info, ir, ir_stream);
    xlr8_read_pr_vec(info, pr, pr_stream);
    xlr8_read_in_vec(x_size, x, x_stream);
    xlr8_vec_load_padded(info, jc_stream, ir_stream, pr_stream, cols_lens_stream, pr_padded_stream, ir_padded_stream);
    spmv_vec_compute_pad(info, cols_lens_stream, ir_padded_stream, pr_padded_stream, x_stream, y_stream, gbuff);
    spmv_vec_writeback(info, y_stream, y, a, new_vec);
}
