#include <stdio.h>
#include "ldl_ltsolve.h"

void lt_copy_streams(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE B[], hls::stream<int>& jc_stream, hls::stream<int>& ir_stream, hls::stream<DTYPE>& pr_stream, hls::stream<DTYPE>& B_stream) {
    for (int i = 0; i < info.nnz; i++) {
        #pragma HLS PIPELINE
        pr_stream.write(pr[i]);
        ir_stream.write(ir[i]);
    }

    for (int i = 0; i < info.n; i++) {
        #pragma HLS PIPELINE
        jc_stream.write(jc[i]);
        B_stream.write(B[i]);
    }
    
    jc_stream.write(jc[info.n]);    
}

void ltsolve(spm_info info, hls::stream<int>& jc_stream, hls::stream<int>& ir_stream, hls::stream<DTYPE>& pr_stream, hls::stream<DTYPE>& B_stream, hls::stream<DTYPE>& X_stream, DTYPE* buffer) {
    
    // Assuming that all the matrix streams are reversed: X_stream, B_stream, jc_stream, ir_stream, pr_stream

    /* Some possible optimizations:
        1. fill B_stream in correct order and change the reading loop s.t. it can be parallelized a little (until first RAW inside second for)
        2. The cycles always read the last 1 of the diagonal, not needed. before the external for throw that 1 with it ir and jc, then put j = 1 in external for (in fact it always skip the j = 0 iteration)
    */

    /*
    // First version read
    int j, p, col_start, col_end, col_index;
    INIT_X_BUFFER: for( j=0; j < info.n; j++ )
    { 
        #pragma HLS PIPELINE
        buffer[j] = B_stream.read(); 
    }
    */

    // Second version read
    int j, p, col_start, col_end, col_index;
    DTYPE buffer_p, pr;
    INIT_X_BUFFER: for( j=info.n-1; j >= 0; j-- )
    { 
        #pragma HLS PIPELINE
        buffer[j] = B_stream.read(); 
    }

    /*
    This is the version with 1 on the diagonal 

    col_start = jc_stream.read(); // end of last column (row when T)
    EACH_COLUMN: for (j = 0; j < info.n; j++){	

        col_end = col_start;	// shift register?
		col_start   = jc_stream.read();
		ALL_ELEMENTS_INSIDE: for (p = col_end - 1; p > col_start ; p--){ // just a counter, -1 since I don't want to consider the 1 on each diagonal

            col_index = ir_stream.read();
			buffer[j] -= pr_stream.read() * buffer[info.n - col_index - 1]; 
		}

        // The information about the 1 on the current diagonal could be thrown
        ir_stream.read(); 
        pr_stream.read();
    }
    */

    /*
    // This is the version without the 1 on the diagonal, first version read
    col_start = jc_stream.read(); // end of last column (row when T)
    EACH_COLUMN: for (j = 0; j < info.n; j++){	
        #pragma HLS PIPELINE

        col_end = col_start;	// shift register?
		col_start   = jc_stream.read();
		ALL_ELEMENTS_INSIDE: for (p = col_end; p > col_start ; p--){ // just a counter, -1 since I don't want to consider the 1 on each diagonal
            #pragma HLS PIPELINE II=10

            col_index = ir_stream.read();
			buffer[j] -= pr_stream.read() * buffer[info.n - col_index - 1]; 
		}
    }
    */
    // Second version read
    col_start = jc_stream.read(); // end of last column (row when T)
    EACH_COLUMN: for (j = info.n-1; j >= 0; j--){	
        #pragma HLS PIPELINE

        col_end = col_start;	// shift register?
        col_start   = jc_stream.read();
        ALL_ELEMENTS_INSIDE: for (p = col_end; p > col_start ; p--){ // just a counter, -1 since I don't want to consider the 1 on each diagonal
            #pragma HLS PIPELINE II=10

            col_index = ir_stream.read();
            buffer_p = buffer[col_index];
            pr = pr_stream.read();
            buffer[j] -=  pr * buffer_p; 
            printf("%f * %f = %f", pr, buffer_p, buffer[j]);
        }
    }

    /*
    // First version read
    WRITE_X_STREAM: for(j=info.n - 1; j >= 0 ; j--)
    { 
        #pragma HLS PIPELINE
        X_stream.write(buffer[j]);
    }
    */
    // Second version read
    WRITE_X_STREAM: for(j=0; j < info.n ; j++)
    { 
        #pragma HLS PIPELINE
        X_stream.write(buffer[j]);
    }

}

void lt_writeback(spm_info info, hls::stream<DTYPE>& X_stream, DTYPE X[]) {
    for (int i = 0; i < info.n; i++) {
        #pragma HLS PIPELINE
        DTYPE value = X_stream.read();
        X[i] = value;
    }
}

void ldl_ltsolve(spm_info info, int jc[], int ir[], DTYPE pr[], DTYPE B[], DTYPE X[]) {
    #pragma HLS INTERFACE mode=m_axi bundle=jc_maxi depth=MAX_JC port=jc offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=ir_maxi depth=MAX_NNZ port=ir offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=pr_maxi depth=MAX_NNZ port=pr offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=X_maxi depth=MAX_IN_VEC port=X offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=B_maxi depth=MAX_OUT_VEC port=B offset=slave

    #pragma HLS INTERFACE s_axilite port=info bundle=control
    #pragma HLS INTERFACE s_axilite port=jc bundle=control
    #pragma HLS INTERFACE s_axilite port=ir bundle=control
    #pragma HLS INTERFACE s_axilite port=pr bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=X bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    DTYPE buffer[G_BUFF_SIZE] = {0};
    hls::stream<int> jc_stream, ir_stream;
    hls::stream<DTYPE> pr_stream, X_stream, B_stream;

    #pragma HLS STREAM variable=jc_stream depth=MAX_JC
    #pragma HLS STREAM variable=ir_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=pr_stream depth=MAX_NNZ 
    #pragma HLS STREAM variable=B_stream  depth=MAX_IN_VEC 
    #pragma HLS STREAM variable=X_stream  depth=MAX_OUT_VEC

    lt_copy_streams(info, jc, ir, pr, B, jc_stream, ir_stream, pr_stream, B_stream);
    ltsolve(info, jc_stream, ir_stream, pr_stream, B_stream, X_stream, buffer);
    lt_writeback(info, X_stream, X);
}