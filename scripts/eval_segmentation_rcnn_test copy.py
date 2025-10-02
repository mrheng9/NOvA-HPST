import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

sys.path.append('..')
sys.path.append('../../')

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
torch.jit.script = lambda x: x

import re 
import h5py
from glob import glob
from functools import partial
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from scipy.spatial.distance import pdist, cdist
from scipy import stats

from hpst.trainers.masked_rcnn_sim_trainer import MaskedRNNSimTrainer
from hpst.trainers.masked_rcnn_trainer import MaskedRNNTrainer
from hpst.dataset.heterogenous_sparse_sim_dataset import HeterogenousSparseDataset
from hpst.utils.options import Options


# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cmasher as cmr
import seaborn as sb

from torch_scatter import scatter
from scipy.optimize import linear_sum_assignment

from pathlib import Path

CUDA = True
CUDA_DEVICE = 0
USE_TEX = False
TESTING_FILE = None

# TESTING_SOURCE = "interpretation"
# GRADIENT = False

# TESTING_SOURCE = "production"
# GRADIENT = False

TESTING_SOURCE= "testing"
GRADIENT = False

NETWORK = MaskedRNNTrainer

TESTING_FILE = "/baldig/physicsprojects2/dikshans/datasets/nova/preprocessed_nova_miniprod6_1_cvnlabmaps.h5"
BASE_DIRECTORY =  "../results/mask_rcnn/lightning_logs/version_263"
OPTIONS_PATH = "/home/dikshans/hpst/config/rcnn/rcnn_tune.json"
CHECKPOINT_PATH = "/home/dikshans/hpst/HPST/0t5rigjs/checkpoints/last.ckpt"

# Load checkpoint and add the test file location
options = Options.load(OPTIONS_PATH)

if TESTING_FILE:
    options.testing_file = TESTING_FILE
else:
    options.testing_file = options.training_file.replace("training", TESTING_SOURCE)
options.num_dataloader_workers = 0

options.training_file = TESTING_FILE

if CHECKPOINT_PATH is None:
    checkpoints = glob(f"{BASE_DIRECTORY}/checkpoints/epoch*.ckpt")
    last_checkpoint = np.argmax([int(re.search("step=(.*).ckpt", s)[1]) for s in checkpoints])
    checkpoint_path = checkpoints[last_checkpoint]
else:
    checkpoint_path = CHECKPOINT_PATH
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint["state_dict"]
print(f"Loading from: {checkpoint_path}")

network = NETWORK(options)
network.load_state_dict(state_dict)

network = network.eval()

if not GRADIENT:
    for parameter in network.parameters():
        parameter.requires_grad_(False)

if CUDA:
    network = network.cuda(CUDA_DEVICE)



DATASET = network.validation_dataset
DATASET.return_index = True

dataloader_options = network.dataloader_options
dataloader_options["pin_memory"] = False
dataloader_options["num_workers"] = 0
dataloader_options["batch_size"] = 220
dataloader_options["drop_last"] = False

test_dataloader = network.dataloader(DATASET, **dataloader_options)


# sum_efficiencies = [0]*8
# n_efficiencies = [0]*8
# sum_purities = [0]*8
# n_purities = [0]*8

# n_corr = 0
# n_tot = 0
# for batch_idx, batch in enumerate(tqdm(test_dataloader)):
#     if CUDA:
#         (ids, features_x, targets_x, features_y, targets_y) = batch
#         ids = ids.cuda(CUDA_DEVICE)
#         features_x = [b.cuda(CUDA_DEVICE) for b in features_x]
#         targets_x = [{k: b[k].cuda(CUDA_DEVICE) for k in b} for b in targets_x]

#         features_y = [b.cuda(CUDA_DEVICE) for b in features_y]
#         targets_y = [{k: b[k].cuda(CUDA_DEVICE) for k in b} for b in targets_y]
    
#     output1, output2 = network.forward(features_x, features_y, targets_x, targets_y)

#     total_coords1 = []
#     total_coords2 = []
    
#     total_predictions1 = []
#     total_predictions2 = []
    
#     total_object_predictions1 = []
#     total_object_predictions2 = []
    
    
#     total_targets1 = []
#     total_targets2 = []
    
#     total_object_targets1 = []
#     total_object_targets2 = []
    
#     total_scores1 = []
#     total_scores2 = []
    
#     total_batch_idx1 = []
#     total_batch_idx2 = []
    
#     current_batch1 = 0
#     current_batch2 = 0
    
    
#     for iter_, (o, t) in enumerate(zip(output1, targets_x)):
#         #print(o["masks"])
#         true_object_labels = torch.argmax(t["masks"], axis=0)
#         mask = true_object_labels != 0
#         coords = torch.nonzero(mask)
#         true_object_labels = true_object_labels[mask]
#         true_labels = t["labels"][true_object_labels]
#         if o["masks"].shape[0] > 0:
#             object_predictions = torch.argmax(o["masks"], axis=0)[0, mask]
#             predictions = o["labels"][object_predictions]

#             dist = (o["masks"][:, 0, mask] * o["scores"].unsqueeze(1))
#             ds = dist.sum(axis=0)
#             dist[:, ds != 0] = dist[:, ds != 0] / ds[ds != 0].unsqueeze(0)
            
#             scores = torch.zeros((dist.shape[-1], 6), device=dist.device)
#             for i, l in enumerate(o["labels"]):
#                 scores[:, l] += dist[i]
#             scores[ds == 0, 0] = 1.0
#         else:
#             predictions = torch.zeros_like(true_object_labels)
#             object_predictions = torch.zeros_like(true_object_labels)
#             scores = torch.zeros((true_object_labels.shape[-1], 6), device=true_object_labels.device)
#             scores[:, 0] = 1.0

#         batch = iter_*torch.ones_like(true_object_labels)

#         total_coords1.append(coords)
    
#         total_predictions1.append(predictions)
    
#         total_object_predictions1.append(object_predictions)
    
#         total_targets1.append(true_labels)
    
#         total_object_targets1.append(true_object_labels)
    
#         total_batch_idx1.append(batch)

#         total_scores1.append(scores)
        
#     current_batch1 += len(output1)

#     for iter_, (o, t) in enumerate(zip(output2, targets_y)):
#         #print(o["masks"])
#         true_object_labels = torch.argmax(t["masks"], axis=0)
#         mask = true_object_labels != 0
#         coords = torch.nonzero(mask)
#         true_object_labels = true_object_labels[mask]
#         true_labels = t["labels"][true_object_labels]
#         if o["masks"].shape[0] > 0:
#             object_predictions = torch.argmax(o["masks"], axis=0)[0, mask]
#             predictions = o["labels"][object_predictions]

#             dist = (o["masks"][:, 0, mask] * o["scores"].unsqueeze(1))
#             ds = dist.sum(axis=0)
#             dist[:, ds != 0] = dist[:, ds != 0] / ds[ds != 0].unsqueeze(0)
            
#             scores = torch.zeros((dist.shape[-1], 6), device=dist.device)
#             for i, l in enumerate(o["labels"]):
#                 scores[:, l] += dist[i]
#             scores[ds == 0, 0] = 1.0
#         else:
#             predictions = torch.zeros_like(true_object_labels)
#             object_predictions = torch.zeros_like(true_object_labels)
#             scores = torch.zeros((true_object_labels.shape[-1], 6), device=true_object_labels.device)
#             scores[:, 0] = 1.0

#         batch = iter_*torch.ones_like(true_object_labels)

#         total_coords2.append(coords)
    
#         total_predictions2.append(predictions)
    
#         total_object_predictions2.append(object_predictions)
    
#         total_targets2.append(true_labels)
    
#         total_object_targets2.append(true_object_labels)
    
#         total_batch_idx2.append(batch)

#         total_scores2.append(scores)
    
#     num_objects = 100

#     predictions1 = torch.cat(total_predictions1 + total_predictions2)
#     targets1 = torch.cat(total_targets1 + total_targets2)
#     batch_idx1 = torch.cat(total_batch_idx1 + total_batch_idx2)
#     coords1 = torch.cat(total_coords1 + total_coords2)
#     object_predictions1 = torch.cat(total_object_predictions1 + total_object_predictions2)
#     object_targets1 = torch.cat(total_object_targets1 + total_object_targets2)
#     scores1 = torch.cat(total_scores1 + total_scores2)
    
#     mask1 = ((targets1 != -1) & (object_targets1 != -1) & (object_targets1 < num_objects))
#     #mask2 = ((targets2 != -1) & (object_targets2 != -1) & (object_targets2 < 10))
    
#     #logits = scores1 #[mask1] #torch.cat((object_predictions1[mask1], object_predictions2[mask2]), dim=0)
#     targets = object_targets1[mask1] #torch.cat((object_targets1[mask1], object_targets2[mask2]), dim=0)
#     batches = batch_idx1[mask1] #torch.cat((batches1[mask1], batches2[mask2]), dim=0)

#     object_preds = object_predictions1[mask1]
#     object_preds = F.one_hot(object_preds, num_classes=num_objects)
    
#     batch_size = batches.max()
#     batch_size = batch_size + 1
    
    

#     pre_reshape_cost_matrix = scatter(object_preds, (batches*num_objects)+targets, dim_size=num_objects*batch_size, reduce="sum", dim=0)
#     n_predictions_per_prong = scatter(object_preds, batches, dim_size=batch_size, reduce="sum", dim=0) # -> (batch_size, 10) this 10 belongs to the column
#     n_true_per_prong = scatter(F.one_hot(targets, num_classes=num_objects), batches, dim_size=batch_size, reduce="sum", dim=0) # -> (batch_size, 10) this 10 belongs to the rows
#     batch_target = scatter(targets, (batches*num_objects)+targets, dim_size=num_objects*batch_size, reduce="mean", dim=0)
    
#     # cost_matrix = cost_matrix.reshape((10, batch_size, -1)).transpose(0,1)
#     cost_matrix = pre_reshape_cost_matrix.reshape((batch_size, num_objects, -1)) # (batch_size, 10, 10)
#     cpu_cm = cost_matrix.detach().cpu().numpy()
#     row_inds, col_inds, indices = [], [], []
#     for i, cm in enumerate(cpu_cm):
#         row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
#         row_ind = torch.from_numpy(row_ind)
#         col_ind = torch.from_numpy(col_ind)
#         index = i*torch.ones_like(row_ind)
        
#         row_inds.append(row_ind)
#         col_inds.append(col_ind)
#         indices.append(index)

#     row_inds = torch.cat(row_inds, dim=0).to(cost_matrix.device)
#     col_inds = torch.cat(col_inds, dim=0).to(cost_matrix.device)
#     indices = torch.cat(indices, dim=0).to(cost_matrix.device)
    
#     effs = cost_matrix[indices, row_inds, col_inds]/n_true_per_prong[indices, row_inds]
#     purs = cost_matrix[indices, row_inds, col_inds]/n_predictions_per_prong[indices, col_inds]
#     n_corr += cost_matrix[indices, row_inds, col_inds].sum()
#     n_tot += n_predictions_per_prong[indices, col_inds].sum()
    
#     for i in range(8):
#         ieffs = effs[batch_target == i]
#         n_efficiencies[i] += torch.isfinite(ieffs).sum()
#         sum_efficiencies[i] += ieffs[torch.isfinite(ieffs)].sum()
#         ipurs = purs[batch_target == i]
#         n_purities[i] += torch.isfinite(ipurs).sum()
#         sum_purities[i] += ipurs[torch.isfinite(ipurs)].sum()


# print("Prong accuracy: %.03f" % (n_corr/n_tot))