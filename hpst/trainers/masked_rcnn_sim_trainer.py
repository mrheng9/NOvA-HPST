import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union

from hpst.utils.options import Options
from hpst.trainers.masked_rcnn_trainer import MaskedRNNTrainer
from hpst.dataset.bbox_mask_sim_dataset_2d import BBoxMaskSimDataset


class MaskedRNNSimTrainer(MaskedRNNTrainer):
    def __init__(self, options: Options, train_perc=None):
        super(MaskedRNNSimTrainer, self).__init__(options, train_perc=train_perc)
        self.num_objects = 32

    @property
    def dataset(self):
        return BBoxMaskSimDataset