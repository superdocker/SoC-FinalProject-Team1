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


