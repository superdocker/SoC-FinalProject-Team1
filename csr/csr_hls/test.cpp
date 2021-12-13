#include <stdio.h>
#include "csr.h"

int main(){
    // Read data
    int csr_row[ROW_SIZE+1];
    int csr_col[NUM_FEATURE];
    int numFeaturePerRow[ROW_SIZE];
    FILE *frow, *fcol;
    frow = fopen("pp_csr_row.dat","r");
    fcol = fopen("pp_csr_col.dat","r");
    for (int i=0; i<=ROW_SIZE; i++){
        fscanf(frow,"%d",&csr_row[i]);
    }
    for (int i=0; i<NUM_FEATURE; i++){
        fscanf(fcol,"%d",&csr_col[i]);
    }
    for (int i=0; i<ROW_SIZE; i++){
        numFeaturePerRow[i] = -1;
    }
    fclose(frow);
    fclose(fcol);

    csr(csr_row, csr_col, numFeaturePerRow);
    return 0;
}
