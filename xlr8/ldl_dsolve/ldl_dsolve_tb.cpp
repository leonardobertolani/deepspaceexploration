#include "ldl_dsolve.h"
#include <stdlib.h>
#include <stdio.h>

void LDL_dsolve
(
    int n,		/* D is n-by-n, where n >= 0 */
    DTYPE X [ ],	/* size n.  right-hand-side on input, soln. on output */
    DTYPE D [ ]	/* input of size n, not modified */
)
{
    int j ;
    for (j = 0 ; j < n ; j++){ X [j] /= D [j]; }
}   

bool v_eq(DTYPE* x1, DTYPE* x2, int len) {
	for (auto i = 0; i < len; i++) {
		//printf("x1 %f, x2 %f\n", x1[i], x2[i]);
		if (abs(x1[i] - x2[i]) > 1e-6) {
			return false;
		}
	}
	return true;
}

int main() {
    
    DTYPE pr[572];
    spm_info info = {569, 569, 569};
    DTYPE B[572];
    DTYPE B_sw[572];
    DTYPE X[572];

    for (int i = 0; i < info.n; i++) {
        pr[i] = i+1;
        B[i] = i;
        B_sw[i] = i;
    }

    hls::vector<double, 4> pr_vec[143];
    hls::vector<double, 4> B_vec[143];
    hls::vector<double, 4> X_vec[143];

    for(int i = 0; i < 143; ++i) {
        pr_vec[i][0] = pr[i*4 + 0];
        pr_vec[i][1] = pr[i*4 + 1];
        pr_vec[i][2] = pr[i*4 + 2];
        pr_vec[i][3] = pr[i*4 + 3];

        B_vec[i][0] = B[i*4 + 0];
        B_vec[i][1] = B[i*4 + 1];
        B_vec[i][2] = B[i*4 + 2];
        B_vec[i][3] = B[i*4 + 3];
        
        X_vec[i][0] = X[i*4 + 0];
        X_vec[i][1] = X[i*4 + 1];
        X_vec[i][2] = X[i*4 + 2];
        X_vec[i][3] = X[i*4 + 3];
    }

    ldl_dsolve(info, pr_vec, X_vec, B_vec);

    for(int i = 0; i < 143; ++i) {        
        X[i*4+0] = X_vec[i][0];
        X[i*4+1] = X_vec[i][1];
        X[i*4+2] = X_vec[i][2];
        X[i*4+3] = X_vec[i][3];
    }

    /*
    for (int i = 0; i < info.n; i++) {
        printf("[%d]: %f\n", i, X[i]);
    }
    */

    LDL_dsolve(info.n, B_sw, pr);

    if(v_eq(X, B_sw, info.m)) {
        std::cout << "Test passed";
        return 0;
    }

    std::cout << "Test NOT passed";
    return 1;
}
