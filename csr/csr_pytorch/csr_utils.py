"""
Janghwan Lee
AIHA 2021
2D Sparse convolution using CSR format
"""
import torch

def csr(shape, feature, coors):
    row = shape[1]
    coors = coors[:,1:]
    """ original csr algorithm
    csr_row = torch.zeros(row+1,dtype=torch.int32)
    csr_col = []
    csr_feature = []
    nnz = 0
    for r in range(row):
        for c in range(0,coors.shape[0]):
            coord = coors[c]
            if r == coord[0]:
                nnz += 1
                csr_col.append(coord[1].item())
                csr_feature.append(feature[c])
        csr_row[r+1] = nnz
    csr_col = torch.tensor(csr_col,dtype=torch.int32)
    """
    # reduced version of csr
    csr_row = torch.zeros(row+1,dtype=torch.int32)
    csr_col = coors[:,0]
    csr_feature = feature
    for r in range(row):
        csr_row[r+1] = csr_row[r]+(coors[:,1]==r).nonzero().shape[0]
        
    return [csr_row, csr_col, csr_feature]

class RowMerger(object):
    def __init__(self, num_row):
        self.num_row = num_row
        self.merged_cols = []
        self.cycle = 0

    def merge_ref(self, csr_row, csr_col):
        for i in range(self.num_row):
            if i==0: # first row
                col_candidates = csr_col[:csr_row[2]].unique()
            elif i==self.num_row-1: # last row
                col_candidates = csr_col[csr_row[-3]:].unique()
            else:
                col_candidates = csr_col[csr_row[i-1]:csr_row[i+2]].unique()
            self.merged_cols.append(col_candidates)

    def comparator(self, a,b,c):
        output = []
        a = torch.cat([a,torch.tensor([-1],device='cuda',dtype=torch.int32)])
        b = torch.cat([b,torch.tensor([-1],device='cuda',dtype=torch.int32)])
        c = torch.cat([c,torch.tensor([-1],device='cuda',dtype=torch.int32)])
        isDone = True if a[0]+b[0]+c[0]==-3 else False
        while not isDone:
            self.cycle += 1 # a cylce for comparasion
            a_ = a[0].item() if a[0]!=-1 else float('inf')
            b_ = b[0].item() if b[0]!=-1 else float('inf')
            c_ = c[0].item() if c[0]!=-1 else float('inf')
            min_val = min(a_,b_,c_)
            output.append(min_val)
            a = a[a!=min_val]
            b = b[b!=min_val]
            c = c[c!=min_val]
            isDone = True if a[0]+b[0]+c[0]==-3 else False
        return torch.tensor(output,dtype=torch.int32,device='cuda')

    def merge(self, csr_row, csr_col):
        self.cycle += 1 # load three registers
        for i in range(self.num_row):
            if i==154: import pdb; pdb.set_trace() 
            if i==0:
                entry0 = torch.empty(0,device='cuda',dtype=torch.int32)
                entry1 = csr_col[:csr_row[1]]
                entry2 = csr_col[csr_row[1]:csr_row[2]]
            elif i==self.num_row-1:
                entry0 = csr_col[csr_row[-3]:csr_row[-2]]
                entry1 = csr_col[csr_row[-2]:csr_row[-1]]
                entry2 = torch.empty(0,device='cuda',dtype=torch.int32)
            else:
                entry0 = csr_col[csr_row[i-1]:csr_row[i]]
                entry1 = csr_col[csr_row[i]:csr_row[i+1]]
                entry2 = csr_col[csr_row[i+1]:csr_row[i+2]]
            col_candidates = self.comparator(entry0, entry1, entry2)
            self.merged_cols.append(col_candidates)

class GenOutCSR(object):
    def __init__(self, num_row):
        self.num_row = num_row
        self.out_csr_row = [0]
        self.out_csr_col = []
        self.out_csr_feature = None
        self.cycle = 0

    def generate(self, merged_cols, out_shape, out_channel):
        for i in range(self.num_row):
            if self.out_csr_row[i]==9233: import pdb; pdb.set_trace() 
            self.cycle += 1
            numAct = len(merged_cols[i])
            merged_col = merged_cols[i]
            if numAct == 0:
                self.out_csr_row.append(self.out_csr_row[i])
            elif numAct == 1:
                point = merged_col[0].item()
                self.out_csr_col+=list(range(max(0,point-1),min(point+1,out_shape[0]-1)+1))
                self.out_csr_row.append(self.out_csr_row[i]+min(point+1,out_shape[0]-1)+1-max(0,point-1))
                self.cycle += 1
            else:
                low = high = merged_col[0]
                nnz = 0
                for a in range(1,numAct):
                    if merged_col[a]-merged_col[a-1] <= 3:
                        high = merged_col[a]
                        self.cycle += 1
                        if a==numAct-1:
                            self.out_csr_col+=list(range(max(0,low-1),min(high+1,out_shape[0]-1)+1))
                            nnz += min(high+1,out_shape[0]-1)-max(0,low-1)+1
                            self.cycle += 1
                    else:
                        self.out_csr_col+=list(range(max(0,low-1),min(high+1,out_shape[0]-1)+1))
                        self.cycle += 1
                        nnz += min(high+1,out_shape[0]-1)-max(0,low-1)+1
                        low = high = merged_col[a]
                        if a==numAct-1:
                            self.out_csr_col+=list(range(max(0,low-1),min(high+1,out_shape[0]-1)+1))
                            self.cycle += 1
                            nnz += min(high+1,out_shape[0]-1)-max(0,low-1)+1
                if torch.is_tensor(nnz): nnz = nnz.item()
                self.out_csr_row.append(self.out_csr_row[i]+nnz)
        self.out_csr_feature = torch.zeros([len(self.out_csr_col),out_channel],device='cuda')
