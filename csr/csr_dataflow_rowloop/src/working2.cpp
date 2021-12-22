#include "hls_vector.h"
#include "hls_stream.h"
#include "csr.h"
#include <stdio.h>

const bool debug=true;

//============================ Janghwan utils 
extern "C" {
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

void readCsrData(hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow,
                 hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow_,
                 hls::stream<hls::vector<DTYPE,NUM_FEATURE>> &CsrCol,
                 data_t* csr_row,
                 data_t* csr_col){
    // Make stream
    hls::vector<DTYPE, (ROW_SIZE+1)> row_temp;
    hls::vector<DTYPE, NUM_FEATURE> col_temp;
    for (int i=0; i<(ROW_SIZE+1); i++){
        row_temp[i] = csr_row[i];
    }
    for (int i=0; i<(NUM_FEATURE); i++){
        col_temp[i] = csr_col[i];
    }
    CsrRow.write(row_temp);
    CsrRow_.write(row_temp);
    CsrCol.write(col_temp);
}

void prepare_merge(hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &Candidates,
                   hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow,
                   hls::stream<hls::vector<DTYPE,NUM_FEATURE>> &CsrCol){
    hls::vector<DTYPE, (ROW_SIZE+1)> csr_row;
    hls::vector<DTYPE, NUM_FEATURE> csr_col;
    csr_row = CsrRow.read();
    csr_col = CsrCol.read();
    prepare_merge_row_loop: for(int r=0; r<ROW_SIZE; r++){
        hls::vector<DTYPE, COL_SIZE*3> candidate;
        init_entry: for(int i=0; i<(COL_SIZE*3); i++){
            candidate[i] = COL_SIZE;
        }
        if (r==0){
            prepare_merge_loop0_1: for(int i=0; i<(csr_row[r+1]-csr_row[r]); i++){
                candidate[COL_SIZE+i] = csr_col[csr_row[r]+i];
            }
            prepare_merge_loop0_2: for(int i=0; i<(csr_row[r+2]-csr_row[r+1]); i++){
                candidate[COL_SIZE*2+i] = csr_col[csr_row[r+1]+i];
            }
        }
        else if (r==ROW_SIZE-1){
            prepare_merge_loop1_0: for(int i=0; i<(csr_row[r]-csr_row[r-1]); i++){
                candidate[i] = csr_col[csr_row[r-1]+i];
            }
            prepare_merge_loop1_1: for(int i=0; i<(csr_row[r+1]-csr_row[r]); i++){
                candidate[COL_SIZE+i] = csr_col[csr_row[r]+i];
            }
        }
        else {
            prepare_merge_loop2_0: for(int i=0; i<(csr_row[r]-csr_row[r-1]); i++){
                candidate[i] = csr_col[csr_row[r-1]+i];
            }
            prepare_merge_loop2_1: for(int i=0; i<(csr_row[r+1]-csr_row[r]); i++){
                candidate[COL_SIZE+i] = csr_col[csr_row[r]+i];
            }
            prepare_merge_loop2_2: for(int i=0; i<(csr_row[r+2]-csr_row[r+1]); i++){
                candidate[COL_SIZE*2+i] = csr_col[csr_row[r+1]+i];
            }
        }
        Candidates.write(candidate);
    } // end row
}

void merge(hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol,
           hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol_,
           hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &Candidates,
           hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs,
           hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow_){
    hls::vector<DTYPE, (ROW_SIZE+1)> csr_row;
    csr_row = CsrRow_.read();
    for(int r=0; r<ROW_SIZE; r++){
        // merged col in a row
        hls::vector<DTYPE, COL_SIZE> merged_col;
        // load candidates
        hls::vector<DTYPE, COL_SIZE*3> entry = Candidates.read();
        int merge_counter = 0;
        hls::vector<DTYPE, COL_SIZE*3> input_index;
        init_merge: for (int i=0; i<COL_SIZE; i++){
            merged_col[i] = COL_SIZE; // initialize with dummy value
            for (int j=0; j<3; j++){
                input_index[j*COL_SIZE+i] = -1;
            }
        }
        int e0_counter = 0;
        int e1_counter = 0;
        int e2_counter = 0;
        int lowestIdx;
        int lowestVal;
        int kernelIdx;
        int inputIdx;
        merge_loop: for (int i=0; i<COL_SIZE; i++){
            lowestIdx = minIdx(entry[e0_counter],
                               entry[COL_SIZE+e1_counter],
                               entry[COL_SIZE*2+e2_counter]);
            // can be parallelized
            if ((lowestIdx&1)==1){
                kernelIdx = 0;
                inputIdx = e0_counter + csr_row[r-1];
                lowestVal = entry[e0_counter];
                e0_counter++;
                input_index[kernelIdx*COL_SIZE+i] = inputIdx;
            }
            if ((lowestIdx&2)>>1==1){
                kernelIdx = 1;
                inputIdx = e1_counter + csr_row[r];
                lowestVal = entry[COL_SIZE+e1_counter];
                e1_counter++;
                input_index[kernelIdx*COL_SIZE+i] = inputIdx;
            }
            if ((lowestIdx&4)>>2==1){
                kernelIdx = 2;
                inputIdx = e2_counter + csr_row[r+1];
                lowestVal = entry[COL_SIZE*2+e2_counter];
                e2_counter++;
                input_index[kernelIdx*COL_SIZE+i] = inputIdx;
            }
            if (lowestVal==COL_SIZE){
                break;
            }
            merged_col[i] = lowestVal; 
            // If no more value in next index -> skip
        } // end merge_loop
        MergedCol.write(merged_col);
        MergedCol_.write(merged_col);
        KIPairs.write(input_index);
    } // end row
}

void dilate(hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol,
            hls::stream<DTYPE> &OutNumFeaturePerRow,
            hls::stream<DTYPE> &OutCsrCol){
    for(int r=0; r<ROW_SIZE; r++){
        int nnz = 0; // non zero value per row
        // load merged col
        hls::vector<DTYPE, COL_SIZE> merged_col = MergedCol.read();
        // initializae with first two value
        int low = merged_col[0];
        int high = merged_col[0];
        int merged_col_counter = 0;
        if (low==COL_SIZE){ // no value
            printf(" ");
        } else if (merged_col[1]==COL_SIZE) { // one value
            if (low==0){ // left boundary
                OutCsrCol.write(low);
                OutCsrCol.write(low+1);
                nnz += 2;
            } else if (low==COL_SIZE-1) { // right boundary
                OutCsrCol.write(low-1);
                OutCsrCol.write(low);
                nnz += 2;
            } else {
                OutCsrCol.write(low-1);
                OutCsrCol.write(low);
                OutCsrCol.write(low+1);
                nnz += 3;
            }
        } else {
            dilation_loop: for (merged_col_counter=1;
                                merged_col_counter < COL_SIZE-1;
                                merged_col_counter++){
                if (merged_col[merged_col_counter]-merged_col[merged_col_counter-1] <= 3){ // update high only
                    high = merged_col[merged_col_counter];
                    if (merged_col[merged_col_counter+1]==COL_SIZE){
                        dilation_subloop0: for (int i=max(0,low-1);
                                                i<=min(COL_SIZE-1,high+1);
                                                i++) {
                            OutCsrCol.write(i);
                            nnz += 1;
                        }
                        break;
                    }
                } else {
                    dilation_subloop1: for (int i=max(0,low-1);
                                            i<=min(COL_SIZE-1,high+1);
                                            i++) {
                        OutCsrCol.write(i);
                        nnz += 1;
                    }
                    low = merged_col[merged_col_counter];
                    high = merged_col[merged_col_counter];
                    if (merged_col[merged_col_counter+1]==COL_SIZE){
                        dilation_subloop2: for (int i=max(0,low-1);
                                                i<=min(COL_SIZE-1,high+1);
                                                i++) {
                            OutCsrCol.write(i);
                            nnz += 1;
                        }
                        break;
                    }
                }
            } // end each column index
        } // end diation
        OutNumFeaturePerRow.write(nnz);
    } // end row
}

void write_rule(hls::stream<hls::vector<DTYPE,3>> &RuleStream, // k,i,o
                hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol_,
                hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs,
                hls::stream<DTYPE> &OutNumFeaturePerRow,
                hls::stream<DTYPE> &OutCsrCol) {
    int prevIdx = 0;
    int currIdx = OutNumFeaturePerRow.read();
    int nnz = 0;
    hls::vector<DTYPE,3> Rule;
    for(int r=0; r<ROW_SIZE; r++){
        hls::vector<DTYPE, COL_SIZE> merged_col = MergedCol_.read();
        hls::vector<DTYPE, COL_SIZE*3> input_index = KIPairs.read();
        hls::vector<DTYPE, COL_SIZE> out_csr_col;
        //int out_csr_col[COL_SIZE];
        init_out_csr_col: for (int i=0; i<COL_SIZE; i++){
            out_csr_col[i] = -1;
        }
        load_out_csr_col: for (int i=0; i<nnz; i++){
            out_csr_col[i] = OutCsrCol.read();
        }
        int outIdx;
        write_rule_loop: for (int i=0; i<COL_SIZE; i++){
            if (merged_col[i]!=COL_SIZE){
                if (merged_col[i]==0){ // left boundary
                    write_rule_loop0: for (int j=0; j<3; j++){
                        if (input_index[j*COL_SIZE+i]!=-1){ // must be first group
                            //Rule[(j*3+1)*RK+rule_counter[j*3+1]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3+1)*RK+RO+rule_counter[j*3+1]++] = prevIdx;
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx;
                            RuleStream.write(Rule);
                            //Rule[(j*3)*RK+rule_counter[j*3]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3)*RK+RO+rule_counter[j*3]++] = prevIdx+1;
                            Rule[0] = (j*3);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+1;
                            RuleStream.write(Rule);
                        }
                    }
                } else if (merged_col[i]==COL_SIZE-1){ // right boundary
                    write_rule_loop1: for (int j=0; j<3; j++){
                        if (input_index[j*COL_SIZE+i]!=-1){ // must be last group
                            //Rule[(j*3+2)*RK+rule_counter[j*3+2]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3+2)*RK+RO+rule_counter[j*3+2]++] = currIdx-2;
                            Rule[0] = (j*3+2);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = currIdx-2;
                            RuleStream.write(Rule);
                            //Rule[(j*3+1)*RK+rule_counter[j*3+1]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3+1)*RK+RO+rule_counter[j*3+1]++] = currIdx-1;
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = currIdx-1;
                            RuleStream.write(Rule);
                        }
                    }
                } else {
                    write_rule_loop2: for (int j=0; j<3; j++){
                        if (input_index[j*COL_SIZE+i]!=-1){
                            for (int n=0; n<nnz; n++){ // is it neccessary?
                                if (merged_col[i]==out_csr_col[n]){
                                    outIdx = n;
                                }
                            }
                            //Rule[(j*3+2)*RK+rule_counter[j*3+2]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3+2)*RK+RO+rule_counter[j*3+2]++] = prevIdx+outIdx-1;
                            Rule[0] = (j*3+2);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx-1;
                            RuleStream.write(Rule);
                            //Rule[(j*3+1)*RK+rule_counter[j*3+1]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3+1)*RK+RO+rule_counter[j*3+1]++] = prevIdx+outIdx;
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx;
                            RuleStream.write(Rule);
                            //Rule[(j*3)*RK+rule_counter[j*3]] = input_index[j*COL_SIZE+i];
                            //Rule[(j*3)*RK+RO+rule_counter[j*3]++] = prevIdx+outIdx+1;
                            Rule[0] = (j*3);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx+1;
                            RuleStream.write(Rule);
                        }
                    }
                }
            }
        } // end write rule loop
        prevIdx = currIdx;
        if (r!=COL_SIZE-1){
            nnz = OutNumFeaturePerRow.read();
        }
        currIdx = nnz+prevIdx;
    } // end row
    // end signal
    Rule[0] = -1;
    Rule[1] = -1;
    Rule[2] = -1;
    RuleStream.write(Rule);
}

}

extern "C" {
void vadd(data_t* csr_row,
          data_t* csr_col,
          data_t* out_Rule){
#pragma HLS INTERFACE mode=m_axi bundle=m0  port=csr_row
#pragma HLS INTERFACE mode=m_axi bundle=m1  port=csr_col
#pragma HLS INTERFACE mode=m_axi bundle=m2  port=out_Rule

#pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow("CsrRowStream");
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow_("CsrRow2Stream");
    hls::stream<hls::vector<DTYPE,NUM_FEATURE>> CsrCol("CsrColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> Candidates("CandidatesStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol("MergedColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol_("MergedCol2Stream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KIPairs("KIPiarsStream");
    hls::stream<hls::vector<DTYPE,3>> RuleStream("RuleStream"); // kernel, input, output
    hls::stream<DTYPE> OutNumFeaturePerRow("OutNumFeaturePerRowStream");
    hls::stream<DTYPE> OutCsrCol("OutCsrColStream");
    readCsrData(CsrRow, CsrRow_, CsrCol, csr_row, csr_col);

    /* Step 0. Prepare merger
    Input: csr_row, csr_col
    Output: Three candidatess
    */
    prepare_merge(Candidates, CsrRow, CsrCol);

    /* Step 1. Three column merger
    Input: csr_row, csr_col
    Output: Merged cols, Input-Kernel(vertical) pairs
    */
    merge(MergedCol, MergedCol_, Candidates, KIPairs, CsrRow_);

    /* Step 2. Row dilator
    Input: Merged cols
    Output: out_csr_row, out_csr_col
    */
    dilate(MergedCol, OutNumFeaturePerRow, OutCsrCol);

    /* Step 3. Rule generator
    Input: Input-Kernel(vertical) pairs, out_csr_row, out_csr_col
    Output: Rule
    */
    write_rule(RuleStream, MergedCol_, KIPairs, OutNumFeaturePerRow, OutCsrCol);

    hls::vector<DTYPE,3> Rule;

    int RK = NUM_FEATURE*2;
    int RO = NUM_FEATURE;
    int rule_counter[KERNEL_SIZE*KERNEL_SIZE];
    init_rule_cnt: for (int i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++){
        rule_counter[i] = 0;
    }
    init_out_rule: for (int i=0; i<KERNEL_SIZE*KERNEL_SIZE*2*NUM_FEATURE; i++){
        out_Rule[i] = -1;
    }
    while (1){
        Rule = RuleStream.read();
        if (Rule[0]==-1) {
            break;
        } else{
            out_Rule[Rule[0]*RK+rule_counter[Rule[0]]] = Rule[1];
            out_Rule[Rule[0]*RK+RO+rule_counter[Rule[0]]++] = Rule[2];
        }
    }

    if (debug==true){
        int RK = NUM_FEATURE*2;
        int RO = NUM_FEATURE;
        printf("\n\n\n================= Summary =================\n\n");

        printf("[Result] Rule generation done\n");
        for (int i=0; i<9; i++){
            printf("   Kernel %2d  ",i);
        }
        printf("\n");
        for (int i=0; i<9; i++){
            printf(" Input Output ");
        }
        for (int l=0; l<NUM_FEATURE; l++){
            printf("\n");
            for (int k=0; k<KERNEL_SIZE*KERNEL_SIZE; k++){
                printf("|  %4d %4d  |",out_Rule[(8-k)*RK+l],out_Rule[(8-k)*RK+RO+l]);
            }
        }
        printf("\n");
    }
}

}
