import torch
from tqdm import tqdm

class linkedList(object):
    """
    Linked list for open addressing
    """
    def __init__(self, tableSize, extraSize):
        self.tableSize = tableSize
        self.extraSize = extraSize
        self.extraCounter = self.tableSize
        self.table = torch.ones([self.tableSize+extraSize,3],dtype=torch.int32,device='cuda')*-1
        self.checkCycle = True
        self.findCycle = 0
        self.insertCycle = 0
        self.collision = 0
        self.counter = 0

    def flush(self):
        self.extraCounter = self.tableSize
        self.table = torch.ones([self.tableSize+self.extraSize,3],dtype=torch.int32,device='cuda')*-1
        self.checkCycle = True
        self.findCycle = 0
        self.insertCycle = 0

    def __size__(self):
        size = self.table.shape[0]*59 #59 bits per row
        size_byte = int(size)/8
        if size_byte > 10**9:
            print(f'Hash table size: {size_byte/(1024**3):.3f} GB')
        elif size_byte > 10**6:
            print(f'Hash table size: {size_byte/(1024**2):.3f} MB')
        elif size_byte > 10**3:
            print(f'Hash table size: {size_byte/(1024):.3f} KB')

    def rowArrayIdx(self, input, shape):
        b = input[0] * shape[0] * shape[1]
        i1 = input[1] * shape[0]
        i2 = input[2]
    
        return (b+i1+i2).int().item()

    def modular(self, idx):
        idx = idx%self.tableSize
        return int(idx)

    def check(self, idx):
        if idx>=self.tableSize+self.extraSize:
            print('Hash bucket is full')
            raise NotImplementedError 
         # return: [dataEmpty, addrEmpty, addr]
        dataEmpty = self.table[idx][0]==-1
        if dataEmpty:
            return True, True
        else:
            addrEmpty = self.table[idx][1]==-1
            if addrEmpty:
                return False, True
            else:
                return False, False

    def insert(self, addr, data, tag):
        self.table[addr][0] = data
        self.table[addr][2] = tag
        if self.checkCycle:
            self.insertCycle += 1

    def find(self, addr, data, onlyFind=False):
        dataEmpty, addrEmpty = self.check(addr)
        if self.checkCycle:
            self.findCycle += 1
        if dataEmpty:
            # Not exist, return end signal and address
            exist = False
            addr = addr
            end = True
            nextptr = None
        else:
            if self.checkCycle:
                self.findCycle += 1
            if data==self.table[addr][0]:
                # data is in the address, return exist signal and address
                exist = True
                addr = addr
                end = True
                nextptr = None
            else:
                self.collision += 1
                if self.checkCycle:
                    self.findCycle += 1
                if addrEmpty:
                    # data is not in table, return next address to insert
                    # data should be stored in next address (nextptr)
                    if onlyFind:
                        exist = False
                        addr = addr
                        nextptr = None
                        end = True
                    else:
                        exist = False
                        addr = addr
                        nextptr = self.extraCounter
                        self.table[addr][1] = nextptr
                        self.extraCounter += 1
                        end = True
                else:
                    # Next loop with next address
                    exist = False
                    addr = self.table[addr][1].item()
                    end = False
                    nextptr = None
        return exist, addr, end, nextptr

def runHash():
    index = 1
    DATA_PATH = '/home/hwanii/sparse_conv/data_210325/'
    
    input_shape = [41,1600,1408]
    #input_shape = [21,800,704]
    #input_shape = [11,400,352]
    #input_shape = [5,200,176]
    coors = torch.load(DATA_PATH+str(index)+'/indices.pt',map_location='cuda')
    coors = coors[coors[:,0]==0]
    #coors = coors[:100]
    llist = linkedList(tableSize=2**14,extraSize=50000)
    for i in tqdm(range(coors.shape[0])):
        coord = coors[i]
        data = llist.rowArrayIdx(coord, input_shape)
        addr = llist.modular(data)
        end = False
        while not end:
            exist, addr, end, nextptr = llist.find(addr, data)
        if not exist:
            if nextptr is not None:
                addr = nextptr
            llist.insert(addr, data, 1)

#    print(f'Number of item needed to add: {coors.shape[0]}')
#    print(f'Total item in hash table: {(llist.table[:,0]>0).nonzero(as_tuple=False).shape[0]}')
#    print(f'Find for {llist.findCycle} cycles, Insert for {llist.insertCycle} cycles')
    origin_table = llist.table[:llist.tableSize]
    numItemsOrigin = (origin_table[:,0]>0).nonzero(as_tuple=False).shape[0]
    print(f'Table size                        : {llist.tableSize}')
    print(f'Total item in original hash table : {numItemsOrigin}')
    print(f'Additional table                  : {llist.extraCounter-llist.tableSize}')
    print(f'Density                           : {numItemsOrigin / llist.tableSize}')
    print(f'Find cycle                        : {llist.findCycle}')
    print(f'Insert cycle                      : {llist.insertCycle}')
    llist.__size__()


if __name__ == "__main__":
    runHash()
