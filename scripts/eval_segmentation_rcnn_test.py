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



total_coords1 = []
total_coords2 = []

total_predictions1 = []
total_predictions2 = []

total_object_predictions1 = []
total_object_predictions2 = []


total_targets1 = []
total_targets2 = []

total_object_targets1 = []
total_object_targets2 = []

total_scores1 = []
total_scores2 = []

total_batch_idx1 = []
total_batch_idx2 = []

current_batch1 = 0
current_batch2 = 0

num_objects = 32

total_ids = []
for batch_idx, batch in enumerate(tqdm(test_dataloader)):
    # if batch_idx == 500:  # Print only the first few batches for demonstration
    #     break
    if CUDA:
        (ids, features_x, targets_x, features_y, targets_y) = batch
        ids = ids.cuda(CUDA_DEVICE)
        features_x = [b.cuda(CUDA_DEVICE) for b in features_x]
        targets_x = [{k: b[k].cuda(CUDA_DEVICE) for k in b} for b in targets_x]

        features_y = [b.cuda(CUDA_DEVICE) for b in features_y]
        targets_y = [{k: b[k].cuda(CUDA_DEVICE) for k in b} for b in targets_y]
        
        #batch = [b.cuda(CUDA_DEVICE) for b in batch]
    
    
    
    # coordinates1 = torch.nn.functional.pad(coordinates1, (0,1), "constant", 0)
    # coordinates1[:,-1] = coordinates1[:,1]
    # coordinates2 = torch.nn.functional.pad(coordinates2, (0,1), "constant", 0)
    # coordinates2[:,-1] = coordinates2[:,1]
    
    output1, output2 = network.forward(features_x, features_y, targets_x, targets_y)

    for iter_, (o, t) in enumerate(zip(output1, targets_x)):
        #print(o["masks"])
        true_object_labels = torch.argmax(t["masks"], axis=0)
        mask = true_object_labels != 0
        coords = torch.nonzero(mask)
        true_object_labels = true_object_labels[mask]
        true_labels = t["labels"][true_object_labels]
        if o["masks"].shape[0] > 0:
            object_predictions = torch.argmax(o["masks"], axis=0)[0, mask]
            predictions = o["labels"][object_predictions]

            dist = (o["masks"][:, 0, mask] * o["scores"].unsqueeze(1))
            ds = dist.sum(axis=0)
            dist[:, ds != 0] = dist[:, ds != 0] / ds[ds != 0].unsqueeze(0)
            
            scores = torch.zeros((dist.shape[-1], 6), device=dist.device)
            for i, l in enumerate(o["labels"]):
                scores[:, l] += dist[i]
            scores[ds == 0, 0] = 1.0
        else:
            predictions = torch.zeros_like(true_object_labels)
            object_predictions = torch.zeros_like(true_object_labels)
            scores = torch.zeros((true_object_labels.shape[-1], 6), device=true_object_labels.device)
            scores[:, 0] = 1.0

        batch = iter_*torch.ones_like(true_object_labels)

        total_coords1.append(coords.cpu())
    
        total_predictions1.append(predictions.cpu())
    
        total_object_predictions1.append(object_predictions.cpu())
    
        total_targets1.append(true_labels.cpu())
    
        total_object_targets1.append(true_object_labels.cpu())
    
        total_batch_idx1.append(batch + current_batch1 + 1)

        total_scores1.append(scores.cpu())
        
    current_batch1 += len(output1)

    for iter_, (o, t) in enumerate(zip(output2, targets_y)):
        #print(o["masks"])
        true_object_labels = torch.argmax(t["masks"], axis=0)
        mask = true_object_labels != 0
        coords = torch.nonzero(mask)
        true_object_labels = true_object_labels[mask]
        true_labels = t["labels"][true_object_labels]
        if o["masks"].shape[0] > 0:
            object_predictions = torch.argmax(o["masks"], axis=0)[0, mask]
            predictions = o["labels"][object_predictions]

            dist = (o["masks"][:, 0, mask] * o["scores"].unsqueeze(1))
            ds = dist.sum(axis=0)
            dist[:, ds != 0] = dist[:, ds != 0] / ds[ds != 0].unsqueeze(0)
            
            scores = torch.zeros((dist.shape[-1], 6), device=dist.device)
            for i, l in enumerate(o["labels"]):
                scores[:, l] += dist[i]
            scores[ds == 0, 0] = 1.0
        else:
            predictions = torch.zeros_like(true_object_labels)
            object_predictions = torch.zeros_like(true_object_labels)
            scores = torch.zeros((true_object_labels.shape[-1], 6), device=true_object_labels.device)
            scores[:, 0] = 1.0

        batch = iter_*torch.ones_like(true_object_labels)

        total_coords2.append(coords.cpu())
    
        total_predictions2.append(predictions.cpu())
    
        total_object_predictions2.append(object_predictions.cpu())
    
        total_targets2.append(true_labels.cpu())
    
        total_object_targets2.append(true_object_labels.cpu())
    
        total_batch_idx2.append(batch + current_batch2 + 1)

        total_scores2.append(scores.cpu())
        
    current_batch2 += len(output2)


total_predictions1 = torch.cat(total_predictions1)
total_targets1 = torch.cat(total_targets1)
total_batch_idx1 = torch.cat(total_batch_idx1)
total_coords1 = torch.cat(total_coords1)
total_object_predictions1 = torch.cat(total_object_predictions1)
total_object_targets1 = torch.cat(total_object_targets1)
total_scores1 = torch.cat(total_scores1)

total_predictions2 = torch.cat(total_predictions2)
total_targets2 = torch.cat(total_targets2)
total_batch_idx2 = torch.cat(total_batch_idx2)
total_coords2 = torch.cat(total_coords2)
total_object_predictions2 = torch.cat(total_object_predictions2)
total_object_targets2 = torch.cat(total_object_targets2)
total_scores2 = torch.cat(total_scores2)


unique = torch.cat([total_targets1[:].unique().long()]).unique()

tp = torch.cat([total_scores1, total_scores2], axis=0)
total_targets = torch.cat([total_targets1, total_targets2], axis=0)
total_predictions = torch.cat([total_predictions1, total_predictions2], axis=0)
total_object_predictions = torch.cat([total_object_predictions1, total_object_predictions2], axis=0)
total_batch_idx = torch.cat([total_batch_idx1, total_batch_idx2], axis=0)

obj_mask = tp[:,0]==0
tp = tp[obj_mask][:,1:]
total_targets = total_targets[obj_mask]
y_score = tp / tp.sum(axis=-1, keepdims=True)
print(y_score.shape)
print(total_targets.unique())
print(f"ROC AUC: {roc_auc_score(total_targets, y_score, multi_class='ovr', average='weighted'):.3f}")
