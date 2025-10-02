import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union

from hpst.utils.options import Options
from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer
from hpst.dataset.heterogenous_sparse_sim_dataset_2d import HeterogenousSparseDataset

@torch.jit.script
def collate_sparse(data: List[Tuple[int, List[Tensor], List[Tensor]]]):
    hits_index = torch.tensor([d[0] for d in data])
    features1 = torch.cat([d[1][0] for d in data])
    coordinates1 = torch.cat([d[1][1] for d in data])
    targets1 = torch.cat([d[1][2] for d in data])
    object_targets1 = torch.cat([d[1][3] for d in data])
    features2 = torch.cat([d[2][0] for d in data])
    coordinates2 = torch.cat([d[2][1] for d in data])
    targets2 = torch.cat([d[2][2] for d in data])
    object_targets2 = torch.cat([d[2][3] for d in data])
    batches1 = torch.cat([i*torch.ones((d[1][0].shape[0],), device=d[1][0].device) for i,d in enumerate(data)]).to(torch.int64)
    batches2 = torch.cat([i*torch.ones((d[2][0].shape[0],), device=d[2][0].device) for i,d in enumerate(data)]).to(torch.int64)
    return (hits_index, batches1, features1, coordinates1, targets1, object_targets1, batches2, features2, coordinates2, targets2, object_targets2)


class HeterogenousPointSetSimTrainer(HeterogenousPointSetTrainer):
    def __init__(self, options: Options, train_perc=None):
        super(HeterogenousPointSetSimTrainer, self).__init__(options, train_perc=train_perc, num_objects=32)

    @property
    def dataset(self):
        return HeterogenousSparseDataset