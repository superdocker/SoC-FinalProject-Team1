"""
Janghwan Lee
AIHA 2021
2D Sparse convolution using CSR format
"""
import torch
torch.manual_seed(777)
import torch.nn as nn
import spconv
import csr_utils
from csr_utils import RowMerger, GenOutCSR

OUTPUT_LOG = True

class SparseConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, shape):
        super().__init__()
        self.net = spconv.SparseConv2d(in_channel, out_channel, kernel_size, 1, padding, bias=None)
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)

def csr_conv(shape, csr_data, weight):
    # Assuming padding=1, stride=1, so output shape will be same with input shape
    row = shape[1]
    out_shape = shape # normal case [w,h]
    csr_row, csr_col, csr_feature = csr_data
    row_merger = RowMerger(row)
    oCSR = GenOutCSR(row)

    import time
    current = time.time()

    # Step 1. Merge rows
    row_merger.merge(csr_row, csr_col)
    print(f'step 1: {time.time()-current:.5f} sec')
    current = time.time()

    # Step 2. Generate output csr
    oCSR.generate(row_merger.merged_cols, out_shape, weight.shape[3])
    print(f'step 2: {time.time()-current:.5f} sec')
    current = time.time()

    # Step 3. Convolution
    """
    Any loop order is available. 
    In this example, for input -> for kernel -> generate partial sums
    """
    in_idx = 0
    for n in range(row):
       for m in csr_col[csr_row[n]:csr_row[n+1]]: # for valid columns
           for r in range(3):
               for s in range(3):
                   outcoor_row = n-s+1
                   outcoor_col = m-r+1
                   if outcoor_row < 0 or outcoor_row >= out_shape[1] or outcoor_col < 0 or outcoor_col >= out_shape[0]:
                       continue
                   else:
                       for out_idx in range(oCSR.out_csr_row[outcoor_row],oCSR.out_csr_row[outcoor_row+1]):
                           if oCSR.out_csr_col[out_idx]==outcoor_col: break
                       pSum = torch.mm(csr_feature[in_idx].unsqueeze(0),weight[s,r,:]).squeeze(0)
                       oCSR.out_csr_feature[out_idx] += pSum
           in_idx+=1
    print(f'step 3: {time.time()-current:.5f} sec')
    current = time.time()
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
    in_channel = 4
    out_channel = 4
    init_weight = torch.rand([3,3,in_channel,out_channel],device='cuda')

#    numAct = 100
#    shape = [32, 32] # num_row, num_col
    # Assuming that input coordinates already sorted by row major
#    randCoors = torch.randperm(n=shape[0]*shape[1],
#                               dtype=torch.int32,
#                               device='cuda')[:numAct].sort()[0]
#    b = torch.zeros([numAct],device='cuda')
#    r = randCoors.sort()[0]//shape[0]
#    c = randCoors.sort()[0]%shape[1]
#    coors = torch.stack([b,r,c]).transpose(0,1) # row, col

    # Prepare input
#    shape = [1600, 1408] # [width, height]
#    coors = torch.load('data1/indices_byz.pt')
    # Assuming that input coordinates already sorted by row major

    # Pointpillars
    shape = [320,280]
    coors = torch.load('coors.pt')
    numAct = coors.shape[0]
    b = torch.zeros([numAct],device='cuda').unsqueeze(0).transpose(0,1)
    coors = torch.cat([b,coors],-1)
    sort_idx = coors[:,1]+coors[:,2]*shape[0]
    coors = coors[sort_idx.sort()[1]]
    num_act = coors.shape[0]
    feature = torch.randn([num_act, in_channel],device='cuda')

    print(f'Problem:\n \
    Input feature widht, height: {shape[0]}, {shape[1]}\n \
    Number of active point: {num_act}\n \
    Input channel: {in_channel}\n \
    Output channel: {out_channel}')
    print(f'\nInput coordinates:\n{coors}\n')

    # CSR
    """We assuming that csr encoding is already done"""
    csr_data = csr_utils.csr(shape,feature,coors) # row, col, feature
    print(csr_data[0])
    print(csr_data[1])
    out_csr_row, out_csr_col, out_csr_feature = csr_conv(shape, csr_data, init_weight)
    if OUTPUT_LOG:
        print(f'===> Generated output csr row:\n{out_csr_row}\n')

    # Reference output
    model = SparseConvLayer(in_channel, out_channel, 3, 1, shape)
    model.state_dict()['net.weight'].copy_(init_weight.data)
    model.cuda()
    output = model(feature, coors, 1)
    if OUTPUT_LOG:
        print(f'===> Generated output csr col:\n{out_csr_col}')

    # Summary 
    print(f'\n===> Reference output sum: {output.features.sum().item()}')
    print(f'===> Generated output sum: {out_csr_feature.sum()}')
    print(f'\n===> Reference output num: {output.features.shape[0]}')
    print(f'===> Generated output num: {len(out_csr_col)}')

    outids = output.indices
    sort_idx = outids[:,1]+outids[:,2]*shape[0]
    outids = outids[sort_idx.sort()[1]]
    out_data = csr_utils.csr(shape,out_csr_feature,outids)
    print(out_data[1])

if __name__ == "__main__":
    main()
