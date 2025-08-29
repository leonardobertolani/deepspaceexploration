#include "xlr8_vec.h"
#include "../spmv_vec/spmv_vec.h"
#include "../spmtvm_vec/spmtvm_vec.h"
#include "../ldl_lsolve_vec/ldl_lsolve_vec.h"
#include "../ldl_dsolve/ldl_dsolve.h"
#include "../ldl_ltsolve_vec/ldl_ltsolve_vec.h"

void xlr8_read_jc_vec(spm_info info, int_vec jc[], hls::stream<int_vec>& jc_stream) {
    #pragma HLS inline off
    int size = (info.n + 1) >> 3;
    if ((info.n + 1) % 8 > 0) {
        size++;
    }
    READ_JC_VEC: for (int i = 0; i < size; i++) {
        #pragma HLS loop_tripcount min=28 max=28
        #pragma HLS PIPELINE
        int_vec v = jc[i];
        jc_stream.write(v);
    }
}

void xlr8_read_ir_vec(spm_info info, int_vec ir[], hls::stream<int_vec>& ir_stream) {    
    #pragma HLS inline off
    int size = info.nnz >> 3;
    if (info.nnz % 8 > 0) {
        size++;
    }
    READ_JC_VEC: for (int i = 0; i < size; i++) {
        #pragma HLS loop_tripcount min=28 max=80
        #pragma HLS PIPELINE
        int_vec v = ir[i];
        ir_stream.write(v);
    }
}

void xlr8_read_pr_vec(spm_info info, double_vec pr[], hls::stream<double_vec>& pr_stream) {
    #pragma HLS inline off
    int size = info.nnz >> 2;
    if (info.nnz % 4 > 0) {
        size++;
    }
    READ_PR_VEC: for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS loop_tripcount min=55 max=160
        double_vec v = pr[i];
        pr_stream.write(v);
    }
}

void xlr8_read_in_vec(int len, double_vec in[], hls::stream<double_vec>& in_stream) {
    #pragma HLS inline off
    READ_IN_VEC: for (int i = 0; i < len; i++) {
        #pragma HLS PIPELINE
        #pragma HLS loop_tripcount min=56 max=56
        double_vec v = in[i];
        in_stream.write(v);
    }
}

void xlr8_vec_load_padded(spm_info info, hls::stream<int_vec>& jc, hls::stream<int_vec>& ir, hls::stream<double_vec>& pr, hls::stream<col_len> &cols_lens, hls::stream<double_vec>& pr_padded, hls::stream<int4_vec>& ir_padded) {    
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
                cl.cl = next_jc[0] - curr_jc[z];
            } else {
                cl.cl = curr_jc[z + 1] - curr_jc[z];
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
                /*PAD_COL: for (int j = 0; j < 4; j++) {
                    #pragma HLS UNROLL off
                    #pragma HLS PIPELINE off
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
                        t_ir_end[j] = info.m + j;
                    }
                }*/

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
}

void xlr8_vec_spmv(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec x[], double_vec y[], flag a, flag new_vec) {
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

void xlr8_vec_spmtvm(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec x[], double_vec y[], flag new_vec, flag skip_diagonal) {
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

void xlr8_vec_lsolve(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]) {
    #pragma HLS DATAFLOW

    DTYPE gbuff[G_BUFF_SIZE] = {0.0};
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

    lsolve_dvec_to_buff(B, gbuff, info.n);
    xlr8_read_jc_vec(info, jc, jc_stream);
    xlr8_read_ir_vec(info, ir, ir_stream);
    xlr8_read_pr_vec(info, pr, pr_stream);
    xlr8_vec_load_padded(info, jc_stream, ir_stream, pr_stream, cols_lens_stream, pr_padded_stream, ir_padded_stream);
    lsolve_vec_compute_pad(info, cols_lens_stream, ir_padded_stream, pr_padded_stream, X_stream, gbuff);
    lsolve_vec_writeback(info, X_stream, X);
}

void xlr8_vec_dsolve(spm_info info, double_vec pr[], double_vec X[], double_vec B[]) {
    #pragma HLS DATAFLOW

    hls::stream<double_vec> pr_stream, X_stream, B_stream;

    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ
    #pragma HLS STREAM variable=X_stream  depth=MAX_OUT_VEC
    #pragma HLS STREAM variable=B_stream  depth=MAX_IN_VEC

    d_copy_streams(info, pr, B, pr_stream, B_stream);
    dsolve(info, pr_stream, B_stream, X_stream);
    d_writeback(info, X_stream, X);    
}

void xlr8_vec_ltsolve(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec X[], double_vec B[]) {
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

void xlr8_vec(fn_code function, spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec in_vec[], double_vec out_vec[], params p) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=72 port=jc offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=122 port=ir offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=243 port=pr offset=slave max_widen_bitwidth=256 
    #pragma HLS INTERFACE mode=m_axi bundle=in_vec_maxi depth=143 port=in_vec offset=slave max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=out_vec_maxi depth=143 port=out_vec offset=slave max_widen_bitwidth=256

    #pragma HLS INTERFACE s_axilite port=function bundle=control
    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=in_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=out_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=p bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS allocation function instances=xlr8_read_in_vec limit=1
    #pragma HLS allocation function instances=xlr8_read_ir_vec limit=1
    #pragma HLS allocation function instances=xlr8_read_jc_vec limit=1
    #pragma HLS allocation function instances=xlr8_read_pr_vec limit=1
    #pragma HLS allocation function instances=xlr8_vec_load_padded limit=1

    #pragma HLS DATAFLOW

    flag f1 = p & 0x1;
    flag f2 = (p >> 1) & 0x1;
    switch (function) {
        case FN_SPMV:
            xlr8_vec_spmv(info, jc, ir, pr, in_vec, out_vec, f1, f2);
            break;
        case FN_SPMTVM:
            xlr8_vec_spmtvm(info, jc, ir, pr, in_vec, out_vec, f1, f2);
            break;
        case FN_LSOLVE:
            xlr8_vec_lsolve(info, jc, ir, pr, out_vec, in_vec);
            break;
        case FN_DSOLVE:
            xlr8_vec_dsolve(info, pr, out_vec, in_vec);
            break;
        case FN_LTSOLVE:
            xlr8_vec_ltsolve(info, jc, ir, pr, out_vec, in_vec);
            break;
        default:
            break;
    }  
}
