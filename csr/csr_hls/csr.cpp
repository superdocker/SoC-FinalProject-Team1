#include <stdio.h>
#include "csr.h"

const bool debug = true;
    
void csr(int csr_row[ROW_SIZE+1],
		 int csr_col[NUM_FEATURE],
		 int numFeaturePerRow[ROW_SIZE]){
    // CSR format test data (simple example)
    // int csr_row[ROW_SIZE+1]        = {0,1,3,3,4,7};
    // int csr_col[NUM_FEATURE]       = {3,0,2,1,0,1,4};
    // int numFeaturePerRow[ROW_SIZE] = {-1,-1,-1,-1,-1}; // initialize

    // Three entires for row merger
    int entry0[COL_SIZE];
    int entry1[COL_SIZE];
    int entry2[COL_SIZE];
    // Output: Rule (2*KERNEL_VOLUME*NUM_FEATURE)
    //       : output CSR (next layer input)
    int Rule[KERNEL_SIZE*KERNEL_SIZE][2][NUM_FEATURE];
    init_rule: for (int k=0; k<KERNEL_SIZE*KERNEL_SIZE; k++){
#pragma HLS UNROLL
        for (int l=0; l<NUM_FEATURE; l++){
            Rule[k][0][l] = -1;
            Rule[k][1][l] = -1;
        }
    }
    int rule_counter[KERNEL_SIZE*KERNEL_SIZE];
    init_rule_cnt: for (int i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++){
#pragma HLS UNROLL
        rule_counter[i] = 0;
    }
    int out_csr_row[ROW_SIZE+1];
    int out_csr_col[ROW_SIZE*COL_SIZE]; // max case
    init_out_csr_row: for (int i=0; i<ROW_SIZE+1; i++){
#pragma HLS UNROLL
        out_csr_row[i] = 0;
    }
    init_out_csr_col: for (int i=0; i<ROW_SIZE*COL_SIZE; i++){
#pragma HLS UNROLL
        out_csr_col[i] = -1;
    }
    int out_csr_counter = 0;

    // Pre-calculating the number of feature for each rows
    // It can be calculated in row merger
    cal_num_feature_per_row: for (int i=0; i<ROW_SIZE; i++){
#pragma HLS UNROLL
        numFeaturePerRow[i] = csr_row[i+1]-csr_row[i];
    }

    // Pipelining in each rows
    row_loop: for(int r=0; r<ROW_SIZE; r++){ // for each rows
        // Initialize with COL_SIZE (any column index could be larger than COL_SIZE)
        //============================  Step 1. Merge rows
        init_entry: for (int i=0; i<COL_SIZE; i++){
#pragma HLS UNROLL
            entry0[i] = COL_SIZE;
            entry1[i] = COL_SIZE;
            entry2[i] = COL_SIZE;
        }
        if (debug==true){
            printf("\nRow %d Start\n",r);
        }
///*
        if (r==0){
            prepare_merge_loop0_1: for(int i=0; i<numFeaturePerRow[r]; i++){
                entry1[i] = csr_col[csr_row[r]+i];
            }
            prepare_merge_loop0_2: for(int i=0; i<numFeaturePerRow[r+1]; i++){
                entry2[i] = csr_col[csr_row[r+1]+i];
            }
        }
        else if (r==ROW_SIZE-1){
            prepare_merge_loop1_0: for(int i=0; i<numFeaturePerRow[r-1]; i++){
                entry0[i] = csr_col[csr_row[r-1]+i];
            }
            prepare_merge_loop1_1: for(int i=0; i<numFeaturePerRow[r]; i++){
                entry1[i] = csr_col[csr_row[r]+i];
            }
        }
        else {
            prepare_merge_loop2_0: for(int i=0; i<numFeaturePerRow[r-1]; i++){
                entry0[i] = csr_col[csr_row[r-1]+i];
            }
            prepare_merge_loop2_1: for(int i=0; i<numFeaturePerRow[r]; i++){
                entry1[i] = csr_col[csr_row[r]+i];
            }
            prepare_merge_loop2_2: for(int i=0; i<numFeaturePerRow[r+1]; i++){
                entry2[i] = csr_col[csr_row[r+1]+i];
            }
        }
//*/

/* version2
        if (r==0){
            prepare_merge_loop0_1: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r]) {
                    break;
                } else {
                    entry1[i] = csr_col[csr_row[r]+i];
                }
            }
            prepare_merge_loop0_2: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r+1]) {
                    break;
                } else {
                    entry2[i] = csr_col[csr_row[r+1]+i];
                }
            }
        }
        else if (r==ROW_SIZE-1){
            prepare_merge_loop1_0: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r-1]) {
                    break;
                } else {
                    entry0[i] = csr_col[csr_row[r-1]+i];
                }
            }
            prepare_merge_loop1_1: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r]) {
                    break;
                } else {
                    entry1[i] = csr_col[csr_row[r]+i];
                }
            }
        }
        else {
            prepare_merge_loop2_0: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r-1]) {
                    break;
                } else {
                    entry0[i] = csr_col[csr_row[r-1]+i];
                }
            }
            prepare_merge_loop2_1: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r]) {
                    break;
                } else {
                    entry1[i] = csr_col[csr_row[r]+i];
                }
            }
            prepare_merge_loop2_2: for(int i=0; i<COL_SIZE; i++){
#pragma HLS unroll
                if (i==numFeaturePerRow[r+1]) {
                    break;
                } else {
                    entry2[i] = csr_col[csr_row[r+1]+i];
                }
            }
        }
*/

        // Merge tree
        // Compare min value iteratively
        int merge_counter = 0;
        int merged_col[COL_SIZE]; // size should be 3*COL_SIZE (dense case)
        int input_index[3][COL_SIZE];
        init_merge: for (int i=0; i<COL_SIZE; i++){
#pragma HLS UNROLL
            merged_col[i] = COL_SIZE; // initialize with dummy value
            for (int j=0; j<3; j++){
#pragma HLS UNROLL
                input_index[j][i] = -1;
            }
        }
        int e0_counter = 0;
        int e1_counter = 0;
        int e2_counter = 0;
        int lowestIdx;
        int lowestVal;
        int kernelIdx;
        int inputIdx;
        // v2: using constant loop count
        merge_loop: for (int i=0; i<COL_SIZE; i++){
            lowestIdx = minIdx(entry0[e0_counter],entry1[e1_counter],entry2[e2_counter]);
            // can be parallelized
            if ((lowestIdx&1)==1){
                kernelIdx = 0;
                inputIdx = e0_counter + csr_row[r-1];
                lowestVal = entry0[e0_counter];
                e0_counter++;
                input_index[kernelIdx][i] = inputIdx;
            }
            if ((lowestIdx&2)>>1==1){
                kernelIdx = 1;
                inputIdx = e1_counter + csr_row[r];
                lowestVal = entry1[e1_counter];
                e1_counter++;
                input_index[kernelIdx][i] = inputIdx;
            }
            if ((lowestIdx&4)>>2==1){
                kernelIdx = 2;
                inputIdx = e2_counter + csr_row[r+1];
                lowestVal = entry2[e2_counter];
                e2_counter++;
                input_index[kernelIdx][i] = inputIdx;
            }
            if (lowestVal==COL_SIZE){
                break;
            }
            merged_col[i] = lowestVal; 
            // If no more value in next index -> skip
        } // end merge_loop
        if (debug==true){
            for (int i=0; i<COL_SIZE; i++){
                for (int j=0; j<3; j++){
                    if (merged_col[i]!=COL_SIZE & input_index[j][i]!=-1){
                        printf("[Merge] Input index %d(Column index %d) by Kernel index %d\n",input_index[j][i],merged_col[i], j);
                    }
                }
            }
        } //============================  End Merge rows -> return merged_col_idx
        //============================  Step 2. Dilate row
        // input: merged_col 
        // calculate dilated kernel -> writing for rule 
        // Get output csr col and nnz per row
        int nnz = 0; // non zero value per row
        int out_csr_col_per_row[COL_SIZE];
        init_out_csr_col_per_row: for (int i=0; i<COL_SIZE; i++){
#pragma HLS UNROLL
            out_csr_col_per_row[i] = -1;
        }
        // initializae with first two value
        int low = merged_col[0];
        int high = merged_col[0];
        int merged_col_counter = 0;
        if (low==COL_SIZE){ // no value
            if (debug==true){
                printf("[Dilation] No active col in a row\n");
            }
        } else if (merged_col[1]==COL_SIZE) { // one value
            if (debug==true){
                printf("[Dilation] Only an active col in a row %d\n",low);
            }
            if (low==0){ // left boundary
                out_csr_col_per_row[nnz++] = low;
                out_csr_col_per_row[nnz++] = low+1;
            } else if (low==COL_SIZE-1) { // right boundary
                out_csr_col_per_row[nnz++] = low-1;
                out_csr_col_per_row[nnz++] = low;
            } else {
                out_csr_col_per_row[nnz++] = low-1;
                out_csr_col_per_row[nnz++] = low;
                out_csr_col_per_row[nnz++] = low+1;
            }
        } else {
            if (debug==true){
                printf("[Dilation] Grouping start\n");
            }
            dilation_loop: for (merged_col_counter=1; merged_col_counter < COL_SIZE-1; merged_col_counter++){
                if (merged_col[merged_col_counter]-merged_col[merged_col_counter-1] <= 3){ // update high only
                    high = merged_col[merged_col_counter];
                    if (merged_col[merged_col_counter+1]==COL_SIZE){
                        if (debug==true){
                            printf("[Dilation] Grouping done %d to %d\n",low,high);
                            }
                        dilation_subloop0: for (int i=max(0,low-1); i<=min(COL_SIZE-1,high+1); i++) {
                            out_csr_col_per_row[nnz++] = i;
                        }
                        break;
                    }
                } else {
                    if (debug==true){
                        printf("[Dilation] Grouping done %d to %d\n",low,high);
                    }
                    dilation_subloop1: for (int i=max(0,low-1); i<=min(COL_SIZE-1,high+1); i++) {
                        out_csr_col_per_row[nnz++] = i;
                    }
                    low = merged_col[merged_col_counter];
                    high = merged_col[merged_col_counter];
                    if (merged_col[merged_col_counter+1]==COL_SIZE){
                        if (debug==true){
                            printf("[Dilation] Grouping done %d to %d\n",low,high);
                        }
                        dilation_subloop2: for (int i=max(0,low-1); i<=min(COL_SIZE-1,high+1); i++) {
                            out_csr_col_per_row[nnz++] = i;
                        }
                        break;
                    }
                }
            } // end each column index
        } // end diation
        // write output csr
        write_out_csr: for (int n=0; n<nnz; n++){
#pragma HLS UNROLL
            out_csr_col[out_csr_counter++] = out_csr_col_per_row[n];
        }
        out_csr_row[r+1] = out_csr_row[r]+nnz;

        // Step 3. Write Rule
        int nnz_rule = 0;
        int outIdx;
        write_rule_loop: for (int i=0; i<COL_SIZE; i++){
#pragma HLS UNROLL
            if (merged_col[i]!=COL_SIZE){
                if (merged_col[i]==0){ // left boundary
                    write_rule_loop0: for (int j=0; j<3; j++){
#pragma HLS UNROLL
                        if (input_index[j][i]!=-1){ // must be first group
                            if (debug==true){
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,1,input_index[j][i], out_csr_row[r]);
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,0,input_index[j][i], out_csr_row[r]+1);
                            }
                            Rule[j*3+1][0][rule_counter[j*3+1]] = input_index[j][i];
                            Rule[j*3+1][1][rule_counter[j*3+1]++] = out_csr_row[r];
                            Rule[j*3][0][rule_counter[j*3]] = input_index[j][i];
                            Rule[j*3][1][rule_counter[j*3]++] = out_csr_row[r]+1;
                        }
                    }
                } else if (merged_col[i]==COL_SIZE-1){ // right boundary
                    write_rule_loop1: for (int j=0; j<3; j++){
#pragma HLS UNROLL
                        if (input_index[j][i]!=-1){ // must be last group
                            if (debug==true){
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,2,input_index[j][i], out_csr_row[r+1]-2);
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,1,input_index[j][i], out_csr_row[r+1]-1);
                            }
                            Rule[j*3+2][0][rule_counter[j*3+2]] = input_index[j][i];
                            Rule[j*3+2][1][rule_counter[j*3+2]++] = out_csr_row[r+1]-2;
                            Rule[j*3+1][0][rule_counter[j*3+1]] = input_index[j][i];
                            Rule[j*3+1][1][rule_counter[j*3+1]++] = out_csr_row[r+1]-1;
                        }
                    }
                } else {
                    write_rule_loop2: for (int j=0; j<3; j++){
#pragma HLS UNROLL
                        if (input_index[j][i]!=-1){
                            for (int n=0; n<nnz; n++){ // is it neccessary?
                                if (merged_col[i]==out_csr_col_per_row[n]){
                                    outIdx = n;
                                }
                            }
                            if (debug==true){
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,2,input_index[j][i],out_csr_row[r]+outIdx-1);
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,1,input_index[j][i],out_csr_row[r]+outIdx);
                                printf("[Rule] Kernel %d(-) and %d(|) makes input %d to output %d\n",j,0,input_index[j][i],out_csr_row[r]+outIdx+1);
                            }
                            Rule[j*3+2][0][rule_counter[j*3+2]] = input_index[j][i];
                            Rule[j*3+2][1][rule_counter[j*3+2]++] = out_csr_row[r]+outIdx-1;
                            Rule[j*3+1][0][rule_counter[j*3+1]] = input_index[j][i];
                            Rule[j*3+1][1][rule_counter[j*3+1]++] = out_csr_row[r]+outIdx;
                            Rule[j*3][0][rule_counter[j*3]] = input_index[j][i];
                            Rule[j*3][1][rule_counter[j*3]++] = out_csr_row[r]+outIdx+1;
                        }
                    }
                }
            }
        }
        
    } // end each output rows

    // Check
    if (debug==true){
        printf("\n\n\n================= Summary =================\n\n");
        printf("[Result] Output CSR generation done\n");
        printf("Output CSR Row\n");
        for (int i=0; i<ROW_SIZE+1; i++){
            printf("%d ",out_csr_row[i]);
        }
        printf("\n");
        printf("Output CSR Col\n");
        for (int i=0; i<ROW_SIZE*COL_SIZE; i++){
            if (out_csr_col[i]==-1){ break;}
            printf("%d ",out_csr_col[i]);
        }
        printf("\n");
        printf("\n");

        printf("[Result] Rule generation done\n");
        for (int i=0; i<9; i++){
            printf("\tKernel %2d\t",i);
        }
        printf("\n");
        for (int i=0; i<9; i++){
            printf("  Input  Output ");
        }
        for (int l=0; l<NUM_FEATURE; l++){
            printf("\n");
            for (int k=0; k<KERNEL_SIZE*KERNEL_SIZE; k++){
                printf("|\t%4d %4d\t|",Rule[8-k][0][l],Rule[8-k][1][l]);
            }
        }
    }
}

//============================ Janghwan utils 
int minIdx(int x,int y,int z){
    int min;
    int minIdx;
    if (x <= y) {
        min = x;
        if (x==y){
            minIdx = 3;
        } else{
            minIdx = 1;
        }
    }
    else {
        min = y;
        minIdx = 2;
    }
    if (z <= min){
        if (z==min){
            minIdx = minIdx+4;
        } else{
            min = z;
            minIdx = 4;
        }
    }
    return minIdx; // 8 cases: 000~111
}

int min(int x, int y){
    int ret;
    if (x <= y) {
        ret = x;
    }
    else {
        ret = y;
    }
    return ret;
}

int max(int x, int y){
    int ret;
    if (x >= y) {
        ret = x;
    }
    else {
        ret = y;
    }
    return ret;
}
