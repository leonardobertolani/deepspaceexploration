#include <stdio.h>
#include <hls_vector.h>
#include "ldl_lsolve.h"

void l_copy_streams(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec B[], hls::stream<int_vec>& jc_stream, hls::stream<int_vec>& ir_stream, hls::stream<double_vec>& pr_stream, hls::stream<double_vec>& B_stream) {
    double_vec d_vec1, d_vec2;
    int_vec i_vec;
    int vector_size = (info.nnz + (8 - info.nnz%8))/8;
    for (int i = 0; i < vector_size; i++) {
        #pragma HLS PIPELINE
        i_vec = ir[i];
        d_vec1 = pr[i*2 + 0];
        d_vec2 = pr[i*2 + 1];
        pr_stream.write(d_vec1);
        pr_stream.write(d_vec2);
        ir_stream.write(i_vec);
    }

    // Check this for here because jc actually is n+1, so it may
    // skip the writing of the last value
    vector_size = (info.n + (8 - info.n%8))/8;
    for (int i = 0; i < vector_size; i++) {
        #pragma HLS PIPELINE
        i_vec = jc[i];
        d_vec1 = B[i*2 + 0];
        d_vec2 = B[i*2 + 1];
        B_stream.write(d_vec1);
        B_stream.write(d_vec2);
        jc_stream.write(i_vec);
    }
    
    jc_stream.write(jc[info.n]);    
}

#define VECTOR_SIZE 4
void lsolve(spm_info info, hls::stream<int>& jc_stream, hls::stream<int>& ir_stream, hls::stream<DTYPE>& pr_stream, hls::stream<DTYPE>& B_stream, hls::stream<DTYPE>& X_stream, DTYPE* buffer) {
    #pragma HLS DATAFLOW

    DTYPE pr, buffer_j;
    int j, p, i, col_start, col_end, col_size, row_index;
    INIT_X_BUFFER: for( j=0; j < info.n; j++ )
    { 
        #pragma HLS pipeline
        #pragma HLS LOOP_TRIPCOUNT min=500 max=600 avg=569
        buffer[j] = B_stream.read(); 
    }

    /*
    Version with 1 on diagonal

    col_end = jc_stream.read();
    EACH_COLUMN: for (j = 0 ; j < info.n; j++){	
        col_start = col_end;	// shift register?
		col_end = jc_stream.read();

        // The first reading should be thrown
        ir_stream.read(); 
        pr_stream.read();
		ALL_ELEMENTS_INSIDE: for (p = col_start + 1; p < col_end ; p++){ // just a counter
            row_index = ir_stream.read();
			buffer[row_index] -= pr_stream.read() * buffer[j]; 
		}
    }
    */

    
    // Version without the 1 on the diagonal
    col_end = jc_stream.read();
    EACH_COLUMN: for (j = 0 ; j < info.n; j++){
        #pragma HLS pipeline
        #pragma HLS LOOP_TRIPCOUNT min=500 max=600 avg=569
        col_start = col_end;	// shift register?
		col_end = jc_stream.read();
        //buffer_j = buffer[j];

		ALL_ELEMENTS_INSIDE: for (p = col_start; p < col_end ; p++){ // just a counter
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5 avg=1
            #pragma HLS dependence variable=buffer type=inter false
            row_index = ir_stream.read();
            pr = pr_stream.read();
			buffer[row_index] -= pr * buffer[j]; 
		}
    }
    

    /*
    hls::vector<DTYPE, VECTOR_SIZE> x_vec;
    hls::vector<DTYPE, VECTOR_SIZE> pr_vec;
    hls::vector<DTYPE, VECTOR_SIZE> buffer_vec;
    int ir_vec[VECTOR_SIZE];
    int idx_vec = 0;
    // Version without the 1 on the diagonal, optimization with hls vector
    col_end = jc_stream.read();
    EACH_COLUMN: for (j = 0 ; j < info.n; j++){
        #pragma HLS pipeline
        #pragma HLS LOOP_TRIPCOUNT min=500 max=600 avg=569
        col_start = col_end;	// shift register?
		col_end = jc_stream.read();
        DTYPE buffer_j = buffer[j];

        col_size = col_end - col_start;
        printf("col_size: %d\n", col_size);
        if(col_size >= VECTOR_SIZE) {
            // Vector multiplication
            ALL_ELEMENTS_INSIDE_VEC: for (p = col_start; p < col_end ; p++){ // just a counter
                #pragma HLS pipeline
                #pragma HLS LOOP_TRIPCOUNT min=VECTOR_SIZE max=5 avg=VECTOR_SIZE
                //#pragma HLS dependence variable=buffer type=inter false

                row_index = ir_stream.read();
                pr = pr_stream.read();
                x_vec[idx_vec] = buffer_j;
                pr_vec[idx_vec] = pr;
                buffer_vec[idx_vec] = buffer[row_index];
                ir_vec[idx_vec] = row_index;
                idx_vec++;

                if(idx_vec == VECTOR_SIZE) {
                    x_vec *= pr_vec;
                    buffer_vec -= x_vec;

                    COPY_VEC: for(i = 0; i < VECTOR_SIZE; ++i) {
                        #pragma HLS UNROLL
                        row_index = ir_vec[i];
                        buffer[row_index] = buffer_vec[i];
                    }

                    idx_vec = 0;
                }
            }

            if (idx_vec > 0) {
                x_vec *= pr_vec;
                buffer_vec -= x_vec;

                COPY_TAIL: for (i = 0; i < idx_vec; i++) {
                    #pragma HLS LOOP_FLATTEN off
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=VECTOR_SIZE avg=1
                    int row_index = ir_vec[i];
                    buffer[row_index] = buffer_vec[i];
                }
                idx_vec = 0;
            }

        } else {
            // Too few elements
            ALL_ELEMENTS_INSIDE_NO_VEC: for (p = col_start; p < col_end ; p++){ // just a counter
                #pragma HLS pipeline
                #pragma HLS LOOP_TRIPCOUNT min=1 max=VECTOR_SIZE avg=1
                #pragma HLS dependence variable=buffer type=inter false
                row_index = ir_stream.read();
                buffer[row_index] -= pr_stream.read() * buffer_j; 
            }
        }
		
    }
    */
    

    WRITE_X_STREAM: for( j=0; j < info.n; j++ )
    { 
        #pragma HLS pipeline
        #pragma HLS LOOP_TRIPCOUNT min=500 max=600 avg=569
        X_stream.write(buffer[j]);
    }

}

void l_writeback(spm_info info, hls::stream<DTYPE>& X_stream, DTYPE X[]) {
    for (int i = 0; i < info.n; i++) {
        #pragma HLS PIPELINE
        DTYPE value = X_stream.read();
        X[i] = value;
    }
}

void ldl_lsolve(spm_info info, int_vec jc[], int_vec ir[], double_vec pr[], double_vec B[], double_vec X[]) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=MAX_JC port=jc offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=MAX_NNZ port=ir offset=slave max_read_burst_length=INT_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=MAX_NNZ port=pr offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=X_maxi depth=MAX_IN_VEC port=X offset=slave max_read_burst_length=DTYPE_BURST max_widen_bitwidth=256
    #pragma HLS INTERFACE mode=m_axi bundle=B_maxi depth=MAX_OUT_VEC port=B offset=slave max_read_burst_length=DTYPE_BURST max_write_burst_length=DTYPE_BURST max_widen_bitwidth=256

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=X bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW


    DTYPE buffer[G_BUFF_SIZE];
    hls::stream<int_vec> jc_stream, ir_stream;
    hls::stream<double_vec> pr_stream, X_stream, B_stream;

    #pragma HLS STREAM variable=jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=B_stream  depth=MAX_IN_VEC
    #pragma HLS STREAM variable=X_stream  depth=MAX_OUT_VEC

    l_copy_streams(info, jc, ir, pr, B, jc_stream, ir_stream, pr_stream, B_stream);
    lsolve(info, jc_stream, ir_stream, pr_stream, B_stream, X_stream, buffer);
    l_writeback(info, X_stream, X);
}