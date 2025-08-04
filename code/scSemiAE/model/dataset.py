import torch.utils.data as data
import numpy as np
import torch

"""
Class representing dataset for an single-cell experiment.
"""

class ExperimentDataset(data.Dataset):
    def __init__(self, x, cells, y=[]):
        """
        Parameters
		----------
		x: 
            numpy array of gene expressions of cells (rows are cells)
        cells: list
            cell IDs in the order of appearance
        y: list
            numeric labels of cells (empty list if unknown)
        """
        super(ExperimentDataset, self).__init__()
        self.x = x 
        # the number of cells
        self.nitems = x.shape[0]

        # deal with labeled and unlabeled data respectively
        if len(y)>0:
            print("== Dataset: Found %d items " % x.shape[0])
            print("== Dataset: Found %d classes" % len(np.unique(y)))
            self.y = tuple(y)          
        if len(y)==0:
            y = np.zeros(len(self.x), dtype=np.int64)
            self.y = tuple(y.tolist())

        # convert x to tensor format
        if type(x)==torch.Tensor:
            self.x = x
        else:
            shape = x.shape[1]
            self.x = [torch.from_numpy(inst).view(shape).float() for inst in x]

        self.xIDs = cells
  
    def __getitem__(self, idx):
        return self.x[idx].squeeze(), self.y[idx], self.xIDs[idx]

    def __len__(self):
        return self.nitems
    
    def get_dim(self):
        return self.x[0].shape[0]




    


