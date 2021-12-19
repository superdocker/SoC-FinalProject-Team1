#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "csr.h"
#include <stdio.h>

const bool debug=true;

//============================ Janghwan utils 
extern "C" {
void readCsrData(hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow,
                 hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> &CsrRow_,
                 hls::stream<hls::vector<DTYPE,NUM_FEATURE>> &CsrCol,
                 hls::vector<DTYPE,(ROW_SIZE+1)> &csr_row,
                 hls::vector<DTYPE,(NUM_FEATURE)> &csr_col){
    // Make stream
    hls::vector<DTYPE, (ROW_SIZE+1)> row_temp;
    hls::vector<DTYPE, NUM_FEATURE> col_temp;
    for(int i=0; i<=ROW_SIZE; i++){
        printf("[CSR_ROW] %d\n",csr_row[i]);
    }
    for(int i=0; i<NUM_FEATURE; i++){
        printf("[CSR_COL] %d\n",csr_col[i]);
    }
//    for (int i=0; i<(ROW_SIZE+1); i++){
//        row_temp[i] = csr_row[i];
//    }
//    for (int i=0; i<(NUM_FEATURE); i++){
//        col_temp[i] = csr_col[i];
//    }
//    CsrRow.write(row_temp);
//    CsrRow_.write(row_temp);
//    CsrCol.write(col_temp);
}
}

extern "C" {
//void vadd(hls::vector<DTYPE,(ROW_SIZE+1)> &csr_row,
//          hls::vector<DTYPE,(NUM_FEATURE)> &csr_col,
//          hls::vector<DTYPE,(18*NUM_FEATURE)> *out_Rule){
void vadd(int* csr_row,
          int* csr_col,
          int* out_Rule){

    hls::vector<DTYPE,(ROW_SIZE+1)> in_csr_row;
    hls::vector<DTYPE,(NUM_FEATURE)> in_csr_col;
    for(int i=0; i<=ROW_SIZE; i++){
        in_csr_row[i] = csr_row[i];
    }
    for(int i=0; i<NUM_FEATURE; i++){
        in_csr_col[i] = csr_col[i];
    }

    // Check values
#pragma HLS INTERFACE mode=m_axi bundle=m0  port=csr_row
#pragma HLS INTERFACE mode=m_axi bundle=m1  port=csr_col
#pragma HLS INTERFACE mode=m_axi bundle=m2  port=out_Rule

#pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow("CsrRowStream");
    hls::stream<hls::vector<DTYPE,(ROW_SIZE+1)>> CsrRow_("CsrRow2Stream");
    hls::stream<hls::vector<DTYPE,NUM_FEATURE>> CsrCol("CsrColStream");
    readCsrData(CsrRow, CsrRow_, CsrCol, in_csr_row, in_csr_col);
}

}
