# About codes

## No dataflow optimization 

Codes are in ```csr_onboard_init```  

**src/host.cpp**
```cpp
typedef int data_t;
std::vector<data_t, aligned_allocator<data_t>> csr_row(ROW_SIZE+1);
std::vector<data_t, aligned_allocator<data_t>> csr_col(NUM_FEATURE);
std::vector<data_t, aligned_allocator<data_t>> Rule(KERNEL_SIZE*KERNEL_SIZE*2*NUM_FEATURE);
FILE *frow, *fcol;
frow = fopen("csr_row.dat","r");
fcol = fopen("csr_col.dat","r");
for (int i=0; i<=ROW_SIZE; i++){
    fscanf(frow,"%d",&csr_row[i]);
}
for (int i=0; i<NUM_FEATURE; i++){
    fscanf(fcol,"%d",&csr_col[i]);
}
fclose(frow);
fclose(fcol);
```
Host 파일에서는 CSR data를 파일 형태로 읽은 뒤, main 함수의 argument 형태로 전달합니다.  

**src/vadd.cpp**  
```cpp
void vadd(data_t* in_csr_row, data_t* in_csr_col, data_t* out_Rule){
#pragma HLS INTERFACE m_axi port = in_csr_row bundle = gmem0 depth=600
#pragma HLS INTERFACE m_axi port = in_csr_col bundle = gmem1 depth=600
#pragma HLS INTERFACE m_axi port = out_Rule bundle = gmem2 depth=600
    int csr_row[ROW_SIZE+1];
    int csr_col[NUM_FEATURE];
    int Rule[KERNEL_SIZE*KERNEL_SIZE*2*NUM_FEATURE];
```
Main 함수는 기본적으로 csr_row, csr_col에 해당하는 데이터를 입력으로 받고, 출력으로 Rule을 내보냅니다.  
```cpp
row_loop: for(int r=0; r<ROW_SIZE; r++){ // for each rows
    // Initialize with COL_SIZE (any column index could be larger than COL_SIZE)
    init_entry: for (int i=0; i<COL_SIZE; i++){
        entry0[i] = COL_SIZE;
        entry1[i] = COL_SIZE;
        entry2[i] = COL_SIZE;
    }
...
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
```
제일 먼저 3개의 Row를 Merge해야 하기 때문에 3개의 Row에서 csr_row를 통해 얻은 유효한 non zero column의 갯수만큼 csr_col로부터 읽어옵니다.  
각 entry들은 세 개의 row에 대해서 각각 유효한 column들을 가지고 있습니다.  
```cpp
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
```
Merge loop은 dense한 경우 최대 모든 column들에 대해서 탐색해야합니다. 세 개의 entry에서 가장 작은 column index들을 추출해내가면서 중복 없는 오름차순으로 정렬하는 일을 하게 됩니다. 이 경우 3-way comparator가 필요하고, comparator는 세 개의 값이 들어오면 가장 작은 값이 어느 entry들에서 나오는지 return하는 함수로 다음과 같습니다.   
```cpp
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
```
Merge loop에서는 000 부터 111까지 나오는 신호를 보고 가장 작은 값이 나온 entry의 counter를 업데이트하면서 반복적으로 다음 값에 대해서 탐색하도록 합니다.  
또한 가장 작은 값이 어떤 kernel의 수직 offset에 의해서 output row로 모이게 되었는지 정보를 input_index에 따로 적어주게 됩니다.  
이러한 과정을 모두 반복적으로 시행하게 되면 merged_col에는 중복 없는 오름차순으로 정렬되어 있는 column index들이 모여있게 됩니다.  

```cpp
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
```

Merged 된 Column index들에 대해서, dilation이 일어나서 하나의 연속된 column index로 이어지게 되면 하나의 group으로 볼 수 있습니다. 따라서 merged_col의 column index들을 반복적으로 읽어서, 거리를 잰 뒤 같은 group에 속하는지 판단하고, 각 group이 끝난 경우에 연속적인 column index들 output의 csr_col에 적어주게 됩니다.  

```cpp
...
write_rule_loop2: for (int j=0; j<3; j++){
    if (input_index[j][i]!=-1){
        for (int n=0; n<nnz; n++){ // is it neccessary?
            if (merged_col[i]==out_csr_col_per_row[n]){
                outIdx = n;
            }
        }
        Rule[(j*3+2)*RK+rule_counter[j*3+2]] = input_index[j][i];
        Rule[(j*3+2)*RK+RO+rule_counter[j*3+2]++] = out_csr_row[r]+outIdx-1;
        Rule[(j*3+1)*RK+rule_counter[j*3+1]] = input_index[j][i];
        Rule[(j*3+1)*RK+RO+rule_counter[j*3+1]++] = out_csr_row[r]+outIdx;
        Rule[(j*3)*RK+rule_counter[j*3]] = input_index[j][i];
        Rule[(j*3)*RK+RO+rule_counter[j*3]++] = out_csr_row[r]+outIdx+1;
    }
...
}
``` 

Write rule loop에서는 input index에 적힌 kernel vertical index와 input index를 보고, dilation loop에서 미리 번호가 매겨진 output index에 horizontal한 방향으로 kernel offset을 결정하여 Rule에 바로 써주게 됩니다.

## With dataflow optimization  

Codes are in ```csr_dataflow_rowloop/src```  

```cpp
void vadd(hls::vector<DTYPE,(ROW_SIZE+1)> *csr_row,
          hls::vector<DTYPE,(NUM_FEATURE)> *csr_col,
          hls::vector<DTYPE, 3> *out_Rule){
...
#pragma HLS DATAFLOW
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow("CsrRowStream");
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow_("CsrRow2Stream");
    hls::stream<hls::vector<DTYPE,NUM_FEATURE>> CsrCol("CsrColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> Candidates("CandidatesStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol("MergedColStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> MergedCol_("MergedCol2Stream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KIPairs("KIPiarsStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KIPairs_("KIPiars2Stream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> KStream("KStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> IStream("IStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE*3>> OStream("OStream");
    hls::stream<DTYPE> RuleWriteLength("RuleWriteLengthStream"); // kernel, input, output
    hls::stream<DTYPE> OutNumFeaturePerRow("OutNumFeaturePerRowStream");
    hls::stream<hls::vector<DTYPE,COL_SIZE>> OutCsrCol("OutCsrColStream");
    readCsrData(CsrRow, CsrCol, csr_row, csr_col);
...

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
    Output: Kernel-Input-Output pairs
    */
    write_rule(KStream, IStream, OStream, RuleWriteLength,
               MergedCol_, KIPairs_, OutNumFeaturePerRow, OutCsrCol);

    /* Step 4. Output module
    Input: Kernel-Input-Output pairs
    Output: Rule
    */
    generate_out(KStream, IStream, OStream, RuleWriteLength, out_Rule);
}
```
Main 함수는 다음과 같이 5개의 step으로 나뉘어져 있습니다.  
- Prepare merge
- Merge three rows
- Dilate columns in a row
- Write Rule
- Write back to host  

이러한 모든 과정은 최소 단위의 Stream으로 이어지게 됩니다.   
또한 각 함수들은 모두 Output row에 대한 loop로 이루어져 있습니다.  
Prepare merge 과정은 Dataflow를 적용하지 않았을 때와 동일합니다.   

```cpp
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
    merge_loop: for (int i=0; i<COL_SIZE; i++){
        ...
        // If no more value in next index -> skip
    } // end merge_loop
    MergedCol.write(merged_col);
    KIPairs.write(input_index);
```

Merge loop는 이전에 설명한 것과 동일하지만, Merged column과 Kernel-Input index pair를 Stream방식으로 받아서 넘겨주도록 합니다.  

```cpp
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
                ...
            } else {
                dilation_loop: for (merged_col_counter=1;
                                    merged_col_counter < COL_SIZE-1;
                                    merged_col_counter++){
                    ...
                    }
                } // end each column index
            } // end else
        } // end dilation
        OutCsrCol.write(outcol_temp);       
        OutNumFeaturePerRow.write(nnz);
    } // end row
}
```

Dilate의 큰 변화의 경우에는, dilation은 기본적으로 겹치는 column index들을 고려하여 최종적인 output column index를 결정하는 역할을 합니다.  
기존의 과정에서는 output column을 더 이상 만들 수 없으면 만들어진 데이터들을 다음 write rule stage로 넘어갈 수 있지만, Stream에서는 정해진 size를 미리 알려주는 것이 중요하므로 최대로 (dense한 경우)를 가정한 COL_SIZE만큼의 데이터에 유효한 값들을 써주고, 얼마만큼 읽어야 유효한지 수를 함께 보내줘야합니다.  

```cpp
void write_rule(hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KStream, // k,i,o
                hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &IStream, // k,i,o
                hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &OStream, // k,i,o
                hls::stream<DTYPE> &RuleWriteLength,
                hls::stream<hls::vector<DTYPE,COL_SIZE>> &MergedCol_,
                hls::stream<hls::vector<DTYPE,COL_SIZE*3>> &KIPairs_,
                hls::stream<DTYPE> &OutNumFeaturePerRow,
                hls::stream<hls::vector<DTYPE,COL_SIZE>> &OutCsrCol){
    int prevIdx = 0;
    int currIdx = 0;
    int nnz = 0;
    hls::vector<DTYPE,3> Rule;

    for(int r=0; r<ROW_SIZE; r++){
        // for streaming
        int write_num = 0;
        hls::vector<DTYPE,COL_SIZE*3> k_temp;
        hls::vector<DTYPE,COL_SIZE*3> i_temp;
        hls::vector<DTYPE,COL_SIZE*3> o_temp;

        nnz = OutNumFeaturePerRow.read();
        hls::vector<DTYPE,COL_SIZE> out_csr_col = OutCsrCol.read();
        currIdx = nnz+prevIdx;
        hls::vector<DTYPE, COL_SIZE> merged_col = MergedCol_.read();
        hls::vector<DTYPE, COL_SIZE*3> input_index = KIPairs_.read();
        int outIdx;
        write_rule_loop: for (int i=0; i<COL_SIZE; i++){
            if (merged_col[i]!=COL_SIZE){
                // write_rule
            }
        } // end one row
        KStream.write(k_temp);
        IStream.write(i_temp);
        OStream.write(o_temp);
        RuleWriteLength.write(write_num);
        prevIdx = currIdx;
    } // end whole row
    // end signal
}
```

Write rule에서는 기존에서는 큰 Rule을 통째로 buffer에 들고 있는 상황에서 바로 Kernel offset과 input, output offset을 결정하여 적어줄 수 있었으나, Stream 방식에서는 ```hls::vector``` type이 큰 값을 갖고 있을 수 없으므로 Kernel, Input, Output index를 각각 stream 방식으로 보내주는데, dense한 경우에 최대 COL_SIZE만큼 각각 mapping될 수 있으므로 COL_SIZE만큼씩 크기를 가진 stream으로 선언해줘야 합니다.   
최종적으로 Write Rule 함수는 Kernel, Input, Output pair를 return해주게 되며, 여기서 유효한 mapping의 갯수는 Output Row마다 가변적이기 때문에 몇 개의 pair가 유효한지 알려주는 Stream 역시 필요하게 됩니다.  

최종적으로 Kernel-Input-Output index pair를 유효한 갯수만큼 읽어서 최종적인 Rule에 적어주는 일을 하게 되면 Rule generation이 끝나게 됩니다.  
