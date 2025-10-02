import numpy as np
import torch
from matplotlib import pyplot as plt
# from pytorch_lightning import metrics
import torchmetrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, jit

from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict

from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_scatter import scatter
from scipy.optimize import linear_sum_assignment

from hpst.utils.options import Options
from hpst.trainers.neutrino_base import NeutrinoBase

from hpst.dataset.bbox_mask_sim_dataset import BboxSimMaskDataset
from hpst.models.unet_3d import Unet3d

from hpst.utils.learning_rate_schedules import get_linear_schedule_with_warmup
from hpst.utils.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup

import sys

TArray = np.ndarray

# used to be List[List[Tuple[Tensor, Tensor, Tensor]]]
#@torch.jit.script
#def collate_sparse(data: List[Tuple[int, Tensor, Tensor, Tensor]]):
#    print(data)
#
#   hits_index = torch.tensor([d[0] for d in data])
#    
#    features = torch.stack([d[1] for d in data])
#    targets = torch.stack([d[2] for d in data])
#   object_tartets = torch.stack([d[3] for d in data])
#    
#    return hits_index, features, targets, object_tartets


class Unet3dSimTrainer(NeutrinoBase):
    def __init__(self, options: Options):
        """

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(Unet3dSimTrainer, self).__init__(options)

        self.num_objects = 32
        self.num_classes = self.training_dataset.num_classes
        self.network = Unet3d(num_classes=self.num_classes, num_objects=self.num_objects)
        
        self.beta = self.options.loss_beta
        effective_num = 1.0 - self.beta**self.training_dataset.target_count
        self.weights = (1.0 - self.beta) / effective_num
        self.weights = self.weights / self.weights.sum() * self.training_dataset.num_classes

        self.gamma = options.loss_gamma
        if self.options.loss_beta < 0.01:
            self.beta = 1 - 1 / len(self.training_dataset)

    @property
    def dataset(self):
        return BboxSimMaskDataset
    
    @property
    def dataloader_options(self):
        return {
            "drop_last": False,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers,
            #"collate_fn": collate_sparse
        }

    def forward(self, features) -> Tensor:
        # Normalize the high level layers
        """
        features1 = features1.clone()
        features1 -= self.mean
        features1 /= self.std

        features2 = features2.clone()
        features2 -= self.mean
        features2 /= self.std
        """

        outputs1 = self.network(features)

        return outputs1
    
    def bipartite_loss(self, logits, targets):
        object_preds = -F.log_softmax(logits, dim=-1)
        #TODO: check if this groups the correct dimensions
        #object_preds = object_preds.reshape((object_preds.shape[0], -1, object_preds.shape[-1]))
        object_preds = object_preds.flatten(2,-1)
        targets = targets.reshape((targets.shape[0], -1))
        targets[targets > self.num_objects] = self.num_objects
        targets = F.one_hot(targets, num_classes=self.num_objects + 1)
        targets = targets[...,:self.num_objects]
        #cost_matrix = scatter(object_preds, (batches*self.num_objects)+targets, dim_size=self.num_objects*batch_size, reduce="sum", dim=0)
        #cost_matrix = cost_matrix.reshape((batch_size, self.num_objects, -1))

        
        #print(targets.shape, object_preds.shape)
        cost_matrix = (object_preds.float() @ targets.float())

        cpu_cm = cost_matrix.detach().cpu().numpy()
        row_inds, col_inds, indices = [], [], []
        for i, cm in enumerate(cpu_cm):
            row_ind, col_ind = linear_sum_assignment(cm, maximize=False)
            row_ind = torch.from_numpy(row_ind)
            col_ind = torch.from_numpy(col_ind)
            index = i*torch.ones_like(row_ind)
            row_inds.append(row_ind)
            col_inds.append(col_ind)
            indices.append(index)

        row_inds = torch.cat(row_inds, dim=0).to(cost_matrix.device)
        col_inds = torch.cat(col_inds, dim=0).to(cost_matrix.device)
        indices = torch.cat(indices, dim=0).to(cost_matrix.device)

        return cost_matrix[indices, row_inds, col_inds].sum()/object_preds.shape[0]
    
    def bipartite_accuracy(self, object_preds, targets):
        targets = targets.reshape((targets.shape[0], -1))
        object_preds = object_preds.reshape((object_preds.shape[0], -1))
        
        targets[targets > self.num_objects] = self.num_objects

        targets = F.one_hot(targets, num_classes=self.num_objects + 1)
        object_preds = F.one_hot(object_preds, num_classes=self.num_objects)
        targets = targets[...,:self.num_objects]
        #print(targets.shape, object_preds.shape)
        cost_matrix = (targets.transpose(1,2).float() @ object_preds.float())
        #batch_size = batches.max()
        #batch_size = batch_size + 1
        #cost_matrix = scatter(object_preds, (batches*self.num_objects)+targets, dim_size=self.num_objects*batch_size, reduce="sum", dim=0)
        #cost_matrix = cost_matrix.reshape((batch_size, self.num_objects, -1))
        
        cpu_cm = cost_matrix.detach().cpu().numpy()
        row_inds, col_inds, indices = [], [], []
        for i, cm in enumerate(cpu_cm):
            row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
            row_ind = torch.from_numpy(row_ind)
            col_ind = torch.from_numpy(col_ind)
            index = i*torch.ones_like(row_ind)
            row_inds.append(row_ind)
            col_inds.append(col_ind)
            indices.append(index)

        row_inds = torch.cat(row_inds, dim=0).to(cost_matrix.device)
        col_inds = torch.cat(col_inds, dim=0).to(cost_matrix.device)
        indices = torch.cat(indices, dim=0).to(cost_matrix.device)

        return cost_matrix[indices, row_inds, col_inds].sum()/object_preds.shape[0]

    def training_step(self, batch, batch_idx):
        (_, features1, targets1, object_targets1) = batch

        predictions1, object_predictions1 = self.forward(features1)
        
        #mask1 = ((targets1 != -1)).unsqueeze(1)

        # mask_group = (scatter((~mask).to(int), batch, dim=-1, reduce='sum') == 0)
        # mask = torch.gather(mask_group, 0, batch)

        #predictions1 = predictions1[mask1]
        #object_predictions1 = object_predictions1[mask1]
        #targets1 = targets1[mask1]
        #object_targets1 = object_targets1[mask1]
        #batches1 = batches1[mask1]

        ce_loss1 = F.cross_entropy(predictions1, targets1, weight=self.weights.to(predictions1.device))

        #mask1 = ((object_targets1 != -1) & (object_targets1 < self.num_objects))

        #predictions1 = predictions1[mask1]
        #object_predictions1 = object_predictions1[mask1]
        #targets1 = targets1[mask1]
        #object_targets1 = object_targets1[mask1]
        #batches1 = batches1[mask1]

        object_loss = self.bipartite_loss(
            object_predictions1,
            object_targets1
        )

        # object_loss is averaged over both sets while cross entropy is averaged over each one
        # so it should be roughly half of the size of the cross entropy losses
        loss = object_loss + ce_loss1
        #loss = ce_loss1

        self.log("object_loss", object_loss, batch_size=self.options.batch_size)
        self.log("train_loss", ce_loss1, batch_size=self.options.batch_size)
        self.log("total_train_loss", loss, batch_size=self.options.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        (_, features1, targets1, object_targets1) = batch

        predictions1, object_predictions1 = self.forward(features1)

        # If there's more than self.num_objects objects in the scene, then we ignore the rest
        #mask1 = ((targets1 != -1) & (object_targets1 != -1) & (object_targets1 < self.num_objects)).unsqueeze(0)

        #predictions1 = torch.masked_select(predictions1.transpose(0,1), mask1)
        #object_predictions1 = torch.masked_select(object_predictions1.transpose(0,1), mask1)
        #targets1 = torch.masked_select(targets1.transpose(0,1), mask1)
        #object_targets1 = torch.masked_select(object_targets1.transpose(0,1), mask1)
        
        _, predictions1 = torch.max(predictions1, dim=1)
        _, object_predictions1 = torch.max(object_predictions1, dim=1)

        #print(predictions1.shape, targets1.shape)
        accuracy = (predictions1 == targets1).to(float).mean()
        self.log("val_accuracy", accuracy, sync_dist=True, batch_size=self.options.batch_size)
        object_accuracy = self.bipartite_accuracy(
            object_predictions1,
            object_targets1,
        )
        # ((object_predictions1 == object_targets1).to(float).sum() + (object_predictions2 == object_targets2).to(float).sum())/(len(object_predictions1) + len(object_predictions2))
        self.log("object_accuracy", object_accuracy, sync_dist=True, batch_size=self.options.batch_size)

        return metrics
