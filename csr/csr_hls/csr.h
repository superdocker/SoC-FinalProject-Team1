/*
    Janghwan Lee 
    2021 AIHA
    Title  : 2D sparse convolution using CSR format
    Problem:
    - Kernel size   : 3
    - Stride        : 1 
    - Dilation      : 1
    - Input size    : 5*5
    - Active points : 7
    - Channel dimension is redundant in this example.
*/

const int ROW_SIZE    = 280; // Assume 5x5 2d matrix
const int COL_SIZE    = 320; // Assume 5x5 2d matrix
const int NUM_FEATURE = 4551; // Number of active point
const int KERNEL_SIZE = 3; // Number of active point

void csr(int csr_row[ROW_SIZE+1],
		 int csr_col[NUM_FEATURE],
		 int numFeaturePerRow[ROW_SIZE]);
int minIdx(int x, int y, int z);
int min(int x, int y);
int max(int x, int y);
