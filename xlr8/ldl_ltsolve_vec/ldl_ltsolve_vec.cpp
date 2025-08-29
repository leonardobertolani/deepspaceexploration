#include <stdio.h>
#include <hls_vector.h>
#include "hls_stream.h"
#include "ldl_ltsolve_vec.h"


void ltsolve_vec_load_padded(spm_info info, hls::stream<int_vec>& jc, hls::stream<int_vec>& ir, hls::stream<double_vec>& pr, hls::stream<col_len> &cols_lens, hls::stream<double_vec>& pr_padded, hls::stream<int4_vec>& ir_padded) {    
    
    int_vec curr_jc, next_jc = jc.read();
    col_len cl;
    double_vec curr_pr = pr.read();
    int pr_idx = 0;
    int_vec curr_ir = ir.read();
    int ir_idx = 0;
    int cols_to_read = (info.n + 1) >> 3;
    if ((info.n + 1) % 8 != 0) {
        cols_to_read++;
    }
    int cols_to_write = info.n + (4 - info.n % 4);    
    int cols_written = 0;
    LT_COLUMN_LOOP: for (int i = 0; i < cols_to_read; i++) {
        //#pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=75 min=70 avg=72

        curr_jc = next_jc;
        if (i < cols_to_read - 1) {
            next_jc = jc.read();
        } else {
            next_jc[0] = next_jc[7];
        }
        LT_READ_JC: for (int z = 0; z < 8; z++) {
            if (z == 7) {
                cl.cl = curr_jc[z] - next_jc[0];
            } else {
                cl.cl = curr_jc[z] - curr_jc[z + 1];
            }
            
            ap_uint<8> cl_diff = cl.cl % 4;
            if (cl.cl == 0) {
                cl.cl_padded = 0;
            } else if (cl_diff != 0) {
                cl.cl_padded = cl.cl + (4 - cl_diff);
            } else {
                cl.cl_padded = cl.cl;
            }

            if (cols_written < cols_to_write) {
                cols_lens.write(cl);
                cols_written++;
            }

            int c_to_read = cl.cl >> 2;
            LT_READ_COL: for (int j = 0; j < c_to_read; j++) {
                //#pragma HLS PIPELINE II=4
                #pragma HLS PIPELINE off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=10 avg=1
                double prv[4];
                #pragma HLS ARRAY_PARTITION variable=prv dim=1 type=complete
                int irv[4];
                #pragma HLS ARRAY_PARTITION variable=irv dim=1 type=complete
                double_vec t;
                int4_vec t_ir;
                LT_READ_CVEC: for (int k = 0; k < 4; k++) {
                    #pragma HLS UNROLL off=true
                    #pragma HLS PIPELINE off
                    if (pr_idx == 4) {
                        curr_pr = pr.read();
                        pr_idx = 0;
                    }
                    if (ir_idx == 8) {
                        curr_ir = ir.read();
                        ir_idx = 0;
                    }
                    prv[k] = curr_pr[pr_idx];
                    irv[k] = curr_ir[ir_idx];
                    pr_idx++;
                    ir_idx++;
                }

                t[0] = prv[0];
                t[1] = prv[1];
                t[2] = prv[2];
                t[3] = prv[3];
                t_ir[0] = irv[0];
                t_ir[1] = irv[1];
                t_ir[2] = irv[2];
                t_ir[3] = irv[3];

                pr_padded.write(t);
                ir_padded.write(t_ir);
            }

            int c_rem = cl.cl - (c_to_read << 2);
            if (c_rem > 0) {
                double_vec t_end;
                int4_vec t_ir_end;
                double prv_end[4] = {0.0, 0.0, 0.0, 0.0};
                #pragma HLS ARRAY_PARTITION variable=prv_end dim=1 type=complete
                int irv_end[4] = {0, 0, 0, 0};
                #pragma HLS ARRAY_PARTITION variable=irv_end dim=1 type=complete
                /*
                LT_PAD_COL: for (int j = 0; j < 4; j++) {
                    #pragma HLS UNROLL
                    if (j < c_rem) {
                        if (pr_idx == 4) {
                            curr_pr = pr.read();
                            pr_idx = 0;
                        }
                        if (ir_idx == 8) {
                            curr_ir = ir.read();
                            ir_idx = 0;
                        }
                        t_end[j] = curr_pr[pr_idx];
                        t_ir_end[j] = curr_ir[ir_idx];
                        pr_idx++;
                        ir_idx++;
                    } else {
                        t_end[j] = 0.0;
                        t_ir_end[j] = 0;//info.m + j;
                    }
                }
                */
                LT_PAD_COL: for (int j = 0; j < c_rem; j++) {
                    #pragma HLS PIPELINE off
                    if (pr_idx == 4) {
                        curr_pr = pr.read();
                        pr_idx = 0;
                    }
                    if (ir_idx == 8) {
                        curr_ir = ir.read();
                        ir_idx = 0;
                    }
                    prv_end[j] = curr_pr[pr_idx];
                    irv_end[j] = curr_ir[ir_idx];
                    pr_idx++;
                    ir_idx++;
                }

                t_end[0] = prv_end[0];
                t_end[1] = prv_end[1];
                t_end[2] = prv_end[2];
                t_end[3] = prv_end[3];
                t_ir_end[0] = irv_end[0];
                t_ir_end[1] = irv_end[1];
                t_ir_end[2] = irv_end[2];
                t_ir_end[3] = irv_end[3];

                pr_padded.write(t_end);
                ir_padded.write(t_ir_end);
            }
        }
    }
    
    /*
    #pragma HLS inline off

    int_vec curr_jc, next_jc = jc.read();
    col_len cl;
    double_vec curr_pr = pr.read();
    int pr_idx = 0;
    int_vec curr_ir = ir.read();
    int ir_idx = 0;
    int cols_to_read = (info.n + 1) >> 3;
    if ((info.n + 1) % 8 != 0) {
        cols_to_read++;
    }
    int cols_to_write = info.n + (4 - info.n % 4);    
    int cols_written = 0;
    LOAD_PADDED: for (int i = 0; i < cols_to_read; i++) {
        curr_jc = next_jc;
        if (i < cols_to_read - 1) {
            next_jc = jc.read();
        } else {
            next_jc[0] = next_jc[7];
        }
        READ_JC: for (int z = 0; z < 8; z++) {
            if (z == 7) {
                cl.cl = curr_jc[z] - next_jc[0];
            } else {
                cl.cl = curr_jc[z] - curr_jc[z + 1];
            }
            
            ap_uint<8> cl_diff = cl.cl % 4;
            if (cl.cl == 0) {
                cl.cl_padded = 0;
            } else if (cl_diff != 0) {
                cl.cl_padded = cl.cl + (4 - cl_diff);
            } else {
                cl.cl_padded = cl.cl;
            }

            if (cols_written < cols_to_write) {
                cols_lens.write(cl);
                cols_written++;
            }

            int c_to_read = cl.cl >> 2;
            READ_COL: for (int j = 0; j < c_to_read; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=30
                //#pragma HLS PIPELINE II=4
                #pragma HLS PIPELINE off
                double prv[4];
                #pragma HLS ARRAY_PARTITION variable=prv dim=1 type=complete
                int irv[4];
                #pragma HLS ARRAY_PARTITION variable=irv dim=1 type=complete
                double_vec t;
                int4_vec t_ir;
                READ_CVEC: for (int k = 0; k < 4; k++) {
                    #pragma HLS UNROLL off=true
                    #pragma HLS PIPELINE off
                    if (pr_idx == 4) {
                        curr_pr = pr.read();
                        pr_idx = 0;
                    }
                    if (ir_idx == 8) {
                        curr_ir = ir.read();
                        ir_idx = 0;
                    }
                    prv[k] = curr_pr[pr_idx];
                    irv[k] = curr_ir[ir_idx];
                    pr_idx++;
                    ir_idx++;
                }
                
                t[0] = prv[0];
                t[1] = prv[1];
                t[2] = prv[2];
                t[3] = prv[3];
                t_ir[0] = irv[0];
                t_ir[1] = irv[1];
                t_ir[2] = irv[2];
                t_ir[3] = irv[3];                                

                pr_padded.write(t);
                ir_padded.write(t_ir);
            }

            int c_rem = cl.cl - (c_to_read << 2);
            if (c_rem > 0) {
                double_vec t_end;
                int4_vec t_ir_end;
                double prv_end[4] = {0.0, 0.0, 0.0, 0.0};
                #pragma HLS ARRAY_PARTITION variable=prv_end dim=1 type=complete
                int irv_end[4] = {info.m, info.m + 1, info.m + 2, info.m + 3};
                #pragma HLS ARRAY_PARTITION variable=irv_end dim=1 type=complete
                

                PAD_COL: for (int j = 0; j < c_rem; j++) {
                    #pragma HLS PIPELINE off
                    if (pr_idx == 4) {
                        curr_pr = pr.read();
                        pr_idx = 0;
                    }
                    if (ir_idx == 8) {
                        curr_ir = ir.read();
                        ir_idx = 0;
                    }
                    prv_end[j] = curr_pr[pr_idx];
                    irv_end[j] = curr_ir[ir_idx];
                    pr_idx++;
                    ir_idx++;
                }

                t_end[0] = prv_end[0];
                t_end[1] = prv_end[1];
                t_end[2] = prv_end[2];
                t_end[3] = prv_end[3];
                t_ir_end[0] = irv_end[0];
                t_ir_end[1] = irv_end[1];
                t_ir_end[2] = irv_end[2];
                t_ir_end[3] = irv_end[3];

                pr_padded.write(t_end);
                ir_padded.write(t_ir_end);
            }
        }
    }
    */
}

void ltsolve_vec_compute_pad(spm_info info, hls::stream<col_len>& cols_len, hls::stream<int4_vec>& ir_padded, hls::stream<double_vec>& pr_padded, hls::stream<double_vec>& X, DTYPE* result) {    
    int n_cols = info.n;
    int cols_to_read = n_cols >> 2;
    if (n_cols % 4 != 0) {
        cols_to_read++;
    }

    double_vec x_vec;
    #pragma HLS ARRAY_PARTITION variable=x_vec dim=1 type=complete

    EXTERNAL_LOOP_LTSOLVE: for (int i = 0; i < cols_to_read; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143

        CL_LOOP_LTSOLVE: for (int z = 0; z < 4; z++) {
            col_len cl;
            cl = cols_len.read();

            double temp0, temp1, temp2, temp3;
            int ri = i << 2;
            temp3 = result[ri + z];
            if (cl.cl >= 4) {

                int vec_to_read = cl.cl_padded / 4;
                COMPUTE_COL_LTSOLVE: for (int j = 0; j < vec_to_read; j++) {
                    #pragma HLS PIPELINE II=4
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=10 avg=1
                    //#pragma HLS DEPENDENCE variable=result type=inter true

                    double_vec m_vec = pr_padded.read();
                    int4_vec ir_vec = ir_padded.read();
                    
                    /*
                    COMPUTE_XVEC_LTSOLVE: for (int a = 0; a < 4; a++) {
                        #pragma HLS UNROLL
                        x_vec[a] = result[info.m - 1 - ir_vec[a]];
                    }
                    */
                    
                    
                    int ir_idx0 = ir_vec[0];
                    int ir_idx1 = ir_vec[1];
                    int ir_idx2 = ir_vec[2];
                    int ir_idx3 = ir_vec[3];

                    int result_idx0 = info.m - 1 - ir_idx0;
                    int result_idx1 = info.m - 1 - ir_idx1;
                    int result_idx2 = info.m - 1 - ir_idx2;
                    int result_idx3 = info.m - 1 - ir_idx3;

                    x_vec[0] = result[result_idx0];
                    x_vec[1] = result[result_idx1];
                    x_vec[2] = result[result_idx2];
                    x_vec[3] = result[result_idx3];
                    

                    double_vec t = m_vec * x_vec;

                    /*SUB_LOOP_LTSOLVE: for (int k = 0; k < 4; k++) {
                        #pragma HLS UNROLL
                        temp -= t[k];
                    }*/

                    temp0 = t[0] + t[1];
                    temp1 = t[2] + t[3];
                    temp2 = temp0 + temp1;
                    temp3 = temp3 - temp2;

                }

            } else if (cl.cl > 0){
                double_vec m_vec = pr_padded.read();
                int4_vec ir_vec = ir_padded.read();
                /*
                COMPUTE_SINGLE_LTSOLVE: for (int j = 0; j < 3; j++) {
                    #pragma HLS UNROLL//PIPELINE II=20
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=1           
                    //#pragma HLS DEPENDENCE variable=result type=inter true
                    double x_val = result[info.m - 1 - ir_vec[j]];
                    double t = x_val * m_vec[j];
                    temp -= t;
                }
                */
                /*
                int ir_idx0 = ir_vec[0];
                int ir_idx1 = ir_vec[1];
                int ir_idx2 = ir_vec[2];

                int result_idx0 = info.m - 1 - ir_idx0;
                int result_idx1 = info.m - 1 - ir_idx1;
                int result_idx2 = info.m - 1 - ir_idx2;

                double x_val0 = result[result_idx0];
                double x_val1 = result[result_idx1];
                double x_val2 = result[result_idx2];

                double t0 = x_val0 * m_vec[0];
                double t1 = x_val1 * m_vec[1];
                double t2 = x_val2 * m_vec[2];

                temp0 = t0 + t1;
                temp1 = temp3 - t2;
                temp3 = temp1 - temp0;
                */
                LTSOLVE_COMP_SMALL: for (int k = 0; k < cl.cl; k++) {
                    #pragma HLS PIPELINE
                    //#pragma HLS PIPELINE off
                    int iri = ir_vec[k];
                    int result_idx = info.m - 1 - iri;
                    double x_val = result[result_idx];
                    double t = x_val * m_vec[k];
                    double temp_sub = temp3;
                    temp_sub -= t;
                    temp3 = temp_sub;
                }
                //temp = temp - t0 - t1 - t2;

            }
            result[ri + z] = temp3;
        }
    }
    
    int X_size = info.m >> 2;
    if (info.m % 4 > 0) {
        X_size++;
    }    
    double_vec X_vec;
    #pragma HLS ARRAY_PARTITION variable=X_vec dim=1 type=complete
    WRITE_X_LTSOLVE: for (int i = 0; i < X_size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143
        /*for (int j = 0; j < 4; j++) {
            X_vec[j] = result[i*4 + j];
        }*/
        int xi = i << 2;
        X_vec[0] = result[xi + 0];
        X_vec[1] = result[xi + 1];
        X_vec[2] = result[xi + 2];
        X_vec[3] = result[xi + 3];
        X.write(X_vec);
    }
}

void ltsolve_vec_writeback(spm_info info, hls::stream<double_vec>& X_stream, double_vec X[]) {
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


void ltsolve_vec_dvec_to_buff(double_vec vec[], DTYPE* buff, size_t len) {
    int chunk_number = len >> 2;

    if(len%4 != 0)
        chunk_number++;

    for (size_t i = 0; i < chunk_number; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT max=145 min=140 avg=143
        int bi = i << 2;
        buff[bi + 0] = vec[i][0];
        buff[bi + 1] = vec[i][1];
        buff[bi + 2] = vec[i][2];
        buff[bi + 3] = vec[i][3];
    }
}


void ldl_ltsolve_vec(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]) {
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
    ltsolve_vec_dvec_to_buff(B, gbuff, info.n);
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
    
    ltsolve_vec_load_padded(info, jc_stream, ir_stream, pr_stream, cols_lens_stream, pr_padded_stream, ir_padded_stream);
    ltsolve_vec_compute_pad(info, cols_lens_stream, ir_padded_stream, pr_padded_stream, X_stream, gbuff);
    ltsolve_vec_writeback(info, X_stream, X);
}
