#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "csr.h"

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
                 hls::stream<hls::vector<DTYPE,NUM_FEATURE>> &CsrCol,
                 hls::vector<DTYPE,(ROW_SIZE+1)> *csr_row,
                 hls::vector<DTYPE,(NUM_FEATURE)> *csr_col){
    // Make stream
    hls::vector<DTYPE,(ROW_SIZE+1)> temp_row = csr_row[0];
    hls::vector<DTYPE,(NUM_FEATURE)> temp_col = csr_col[0];
    CsrRow.write(temp_row);
    CsrCol.write(temp_col);
}

void prepare_merge(hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &Candidates,
                   hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow_,
                   hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow,
                   hls::stream<hls::vector<DTYPE,NUM_FEATURE>> &CsrCol){
    hls::vector<DTYPE, (ROW_SIZE+1)> csr_row;
    hls::vector<DTYPE, NUM_FEATURE> csr_col;
    csr_row = CsrRow.read();
    csr_col = CsrCol.read();
    CsrRow_.write(csr_row);
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
        KIPairs.write(input_index);
    } // end row
}

void dilate(hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol_,
            hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs_,
            hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol,
            hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs,
            hls::stream<DTYPE> &OutNumFeaturePerRow,
            hls::stream<hls::vector<DTYPE,COL_SIZE>> &OutCsrCol){
    for(int r=0; r<ROW_SIZE; r++){
        int nnz = 0; // non zero value per row
        // load merged col
        hls::vector<DTYPE, COL_SIZE> merged_col = MergedCol.read();
        hls::vector<DTYPE, COL_SIZE*3> ki_temp = KIPairs.read();
        hls::vector<DTYPE, COL_SIZE> outcol_temp;
        // initialize out_col_csr
        for (int i=0; i<COL_SIZE; i++){
            outcol_temp[i] = -1;
        }
        int outcol_cnt = 0;
        KIPairs_.write(ki_temp);
        MergedCol_.write(merged_col);
        // initializae with first two value
        int low = merged_col[0];
        int high = merged_col[0];
        int merged_col_counter = 0;
        if (low!=COL_SIZE){ // no value
            if (merged_col[1]==COL_SIZE) { // one value
                if (low==0){ // left boundary
                    outcol_temp[outcol_cnt++] = low;
                    outcol_temp[outcol_cnt++] = low+1;
                    nnz += 2;
                } else if (low==COL_SIZE-1) { // right boundary
                    outcol_temp[outcol_cnt++] = low-1;
                    outcol_temp[outcol_cnt++] = low;
                    nnz += 2;
                } else {
                    outcol_temp[outcol_cnt++] = low-1;
                    outcol_temp[outcol_cnt++] = low;
                    outcol_temp[outcol_cnt++] = low+1;
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
                                outcol_temp[outcol_cnt++] = i;
                                nnz += 1;
                            }
                            break;
                        }
                    } else {
                        dilation_subloop1: for (int i=max(0,low-1);
                                                i<=min(COL_SIZE-1,high+1);
                                                i++) {
                            outcol_temp[outcol_cnt++] = i;
                            nnz += 1;
                        }
                        low = merged_col[merged_col_counter];
                        high = merged_col[merged_col_counter];
                        if (merged_col[merged_col_counter+1]==COL_SIZE){
                            dilation_subloop2: for (int i=max(0,low-1);
                                                    i<=min(COL_SIZE-1,high+1);
                                                    i++) {
                                outcol_temp[outcol_cnt++] = i;
                                nnz += 1;
                            }
                            break;
                        }
                    }
                } // end each column index
            } // end else
        } // end dilation
        OutCsrCol.write(outcol_temp);       
        OutNumFeaturePerRow.write(nnz);
    } // end row
}

int write_rule(hls::stream<hls::vector<DTYPE,3>> &RuleStream, // k,i,o
               hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol_,
               hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs_,
               hls::stream<DTYPE> &OutNumFeaturePerRow,
               hls::stream<hls::vector<DTYPE,COL_SIZE>> &OutCsrCol){
    int prevIdx = 0;
    int currIdx = 0;
    int nnz = 0;
    hls::vector<DTYPE,3> Rule;
    int write_num = 0;

    for(int r=0; r<ROW_SIZE; r++){
        nnz = OutNumFeaturePerRow.read();
        hls::vector<DTYPE,COL_SIZE> out_csr_col = OutCsrCol.read();
        currIdx = nnz+prevIdx;
        hls::vector<DTYPE, COL_SIZE> merged_col = MergedCol_.read();
        hls::vector<DTYPE, COL_SIZE*3> input_index = KIPairs_.read();
        int outIdx;
        write_rule_loop: for (int i=0; i<COL_SIZE; i++){
            if (merged_col[i]!=COL_SIZE){
                if (merged_col[i]==0){ // left boundary
                    write_rule_loop0: for (int j=0; j<3; j++){
                        if (input_index[j*COL_SIZE+i]!=-1){ // must be first group
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx;
                            RuleStream.write(Rule);
                            write_num++;
                            Rule[0] = (j*3);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+1;
                            RuleStream.write(Rule);
                            write_num++;
                        }
                    }
                } else if (merged_col[i]==COL_SIZE-1){ // right boundary
                    write_rule_loop1: for (int j=0; j<3; j++){
                        if (input_index[j*COL_SIZE+i]!=-1){ // must be last group
                            Rule[0] = (j*3+2);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = currIdx-2;
                            RuleStream.write(Rule);
                            write_num++;
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = currIdx-1;
                            RuleStream.write(Rule);
                            write_num++;
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
                            Rule[0] = (j*3+2);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx-1;
                            RuleStream.write(Rule);
                            write_num++;
                            Rule[0] = (j*3+1);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx;
                            RuleStream.write(Rule);
                            write_num++;
                            Rule[0] = (j*3);
                            Rule[1] = input_index[j*COL_SIZE+i];
                            Rule[2] = prevIdx+outIdx+1;
                            RuleStream.write(Rule);
                            write_num++;
                        }
                    }
                }
            }
        } // end write rule loop
        prevIdx = currIdx;
    } // end row
    // end signal
    return write_num;
}

}

extern "C" {
void vadd(hls::vector<DTYPE,(ROW_SIZE+1)> *csr_row,
          hls::vector<DTYPE,(NUM_FEATURE)> *csr_col,
          hls::vector<DTYPE, 3> *out_Rule){
#pragma HLS INTERFACE mode=m_axi bundle=gmem0 port=csr_row depth=600
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 port=csr_col depth=600
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 port=out_Rule depth=600

#pragma HLS DATAFLOW
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow("CsrRowStream");
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow_("CsrRow2Stream");
    hls::stream<hls::vector<DTYPE,NUM_FEATURE>> CsrCol("CsrColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> Candidates("CandidatesStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol("MergedColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol_("MergedCol2Stream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KIPairs("KIPiarsStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KIPairs_("KIPiars2Stream");
    hls::stream<hls::vector<DTYPE,3>> RuleStream("RuleStream"); // kernel, input, output
    hls::stream<DTYPE> OutNumFeaturePerRow("OutNumFeaturePerRowStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> OutCsrCol("OutCsrColStream");
    readCsrData(CsrRow, CsrCol, csr_row, csr_col);

    /* Step 0. Prepare merger
    Input: csr_row, csr_col
    Output: Three candidatess
    */
    prepare_merge(Candidates, CsrRow_, CsrRow, CsrCol);

    /* Step 1. Three column merger
    Input: csr_row, csr_col
    Output: Merged cols, Input-Kernel(vertical) pairs
    */
    merge(MergedCol, Candidates, KIPairs, CsrRow_);

    /* Step 2. Row dilator
    Input: Merged cols
    Output: out_csr_row, out_csr_col
    */
    dilate(MergedCol_, KIPairs_, MergedCol, KIPairs, OutNumFeaturePerRow, OutCsrCol);

    /* Step 3. Rule generator
    Input: Input-Kernel(vertical) pairs, out_csr_row, out_csr_col
    Output: Rule
    */
    int rule_size;
    rule_size = write_rule(RuleStream, MergedCol_, KIPairs_, OutNumFeaturePerRow, OutCsrCol);

    // Write out port
    for (int i=0; i<KERNEL_SIZE*KERNEL_SIZE*NUM_FEATURE*3; i++){
        if (i<rule_size){
            hls::vector<DTYPE,3> Rule = RuleStream.read();
            //out_Rule[i] = {Rule[0],Rule[1],Rule[2]};
            out_Rule[i] = Rule;
        } else{
            out_Rule[i] = -1; // for done
        }
    }
}

}
