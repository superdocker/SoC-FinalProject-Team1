"""
Janghwan Lee
AIHA 2021
2D Sparse convolution using CSR format
"""
import torch
import torch.nn as nn
import spconv
import csr_utils
from csr_utils import RowMerger, GenOutCSR
from hashTable import linkedList

import warnings
warnings.simplefilter("ignore", UserWarning)

PRINT_COORDS = False

def using_hash(shape, coors):
    hash_table = linkedList(tableSize=2**14, extraSize=50000) 
    # for all valid points (9*numAct)
    counter = 0
    for i in range(coors.shape[0]):
        for m in [-1,0,1]:
            for n in [-1,0,1]:
                point = coors[i].clone().detach()
                point[1] = point[1]+m
                point[2] = point[2]+n
                if point[1] < 0 or point[1] > shape[0]:
                    continue
                if point[2] < 0 or point[2] > shape[1]:
                    continue
                counter += 1
                data = hash_table.rowArrayIdx(point, shape)
                addr = hash_table.modular(data)
                end = False
                while not end:
                    exist, addr, end, nextptr = hash_table.find(addr, data)
                if exist:
                    outputId = hash_table.table[addr][2]
                else:
                    if nextptr is not None:
                        addr = nextptr
                    hash_table.insert(addr, data, hash_table.counter)
                    outputId = hash_table.counter
                    hash_table.counter += 1
    origin_table = hash_table.table[:hash_table.tableSize]
    numItemsOrigin = (origin_table[:,0]>0).nonzero(as_tuple=False).shape[0]
    print('\n')
    print('Hash table','='*79)
    print('- Summary')
    print(f'Table size                        : {hash_table.tableSize}')
    print(f'Total item in original hash table : {numItemsOrigin}')
    print(f'Additional table                  : {hash_table.extraCounter-hash_table.tableSize}')
    print(f'Total output num                  : {hash_table.insertCycle}')
    print(f'Density                           : {numItemsOrigin / hash_table.tableSize}')
    print('- Cycles')
    print(f'Generate output candidates        : {counter}')
    print(f'Hash find                         : {hash_table.findCycle}')
    print(f'Hash insert                       : {hash_table.insertCycle}')
    print(f'Write Rule                        : {hash_table.insertCycle}')
    print(f'Total cycles                      : {counter+hash_table.findCycle+2*hash_table.insertCycle}')
    print('='*90)

def csr_conv(shape, csr_data, weight):
    # Assuming padding=1, stride=1, so output shape will be same with input shape
    row = shape[1]
    out_shape = shape # normal case [w,h]
    csr_row, csr_col, csr_feature = csr_data
    row_merger = RowMerger(row)
    oCSR = GenOutCSR(row)

    # Step 1. Merge rows
    row_merger.merge(csr_row, csr_col)
    # Step 2. Generate output csr
    oCSR.generate(row_merger.merged_cols, out_shape, weight.shape[3])
    print('\n')
    print('CSR','='*86)
    print('- Summary')
    print(f'Total output num                  : {len(oCSR.out_csr_col)}')
    print('- Cycles')
    print(f'Merge rows                        : {row_merger.cycle}')
    print(f'Generate output CSR               : {oCSR.cycle}')
    print(f'Total cycles                      : {row_merger.cycle+oCSR.cycle}')
    print('='*90)
    return oCSR.out_csr_row, oCSR.out_csr_col, oCSR.out_csr_feature

def main():
    """
    Assuming random input
    1. Size: Size of 2d spatial array
    2. Number of activation: Number of non-zero values
    3. Feature: Random feature which has C channels
    4. Coordinates: Random coordinates of active features
    """
    # Hyperparameter
    in_channel = 16
    out_channel = 32
    init_weight = torch.rand([3,3,in_channel,out_channel],device='cuda')

    # Prepare input
    shape = [1600, 1408] # [width, height]
    coors = torch.load('data1/indices_byz.pt')
    # Assuming that input coordinates already sorted by row major
    sort_idx = coors[:,1]+coors[:,2]*shape[0]
    coors = coors[sort_idx.sort()[1]]
    num_act = coors.shape[0]
    feature = torch.randn([num_act, in_channel],device='cuda')
    print(f'\nProblem:\n \
    Input feature widht, height: {shape[0]}, {shape[1]}\n \
    Number of active point: {num_act}\n \
    Input channel: {in_channel}\n \
    Output channel: {out_channel}')
    if PRINT_COORDS:
        print(f'\nInput coordinates:\n{coors}\n')

    # Hash table
    using_hash(shape, coors)

    # CSR
    """We assuming that csr encoding is already done"""
    csr_data = csr_utils.csr(shape,feature,coors) # row, col, feature
    out_csr_row, out_csr_col, _ = csr_conv(shape, csr_data, init_weight)

if __name__ == "__main__":
    main()
