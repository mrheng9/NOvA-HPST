import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(1, '/home/roblesee/dune/UCI-NuML-Codebase/NOvA/Transformer')

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

from hpst.utils.options import Options
from hpst.trainers.point_set_trainer import PointSetTrainer

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cmasher as cmr
import seaborn as sb

# from scalene import scalene_profiler
from filprofiler.api import profile
import time
from pathlib import Path

CUDA = False
CUDA_DEVICE = 0
USE_TEX = False
TESTING_FILE = None

# TESTING_SOURCE = "interpretation"
# GRADIENT = False

# TESTING_SOURCE = "production"
# GRADIENT = False

TESTING_SOURCE= "testing"
GRADIENT = False

NETWORK = PointSetTrainer

TESTING_FILE = "/baldig/physicsprojects/roblesee/preprocessed_nova_prod5_1_respin_cvnlabmaps.h5"
BASE_DIRECTORY =  "./point_set/lightning_logs/version_15"
CHECKPOINT_PATH = "./point_set/lightning_logs/version_15/checkpoints/last.ckpt"

# Load checkpoint and add the test file location
options = Options.load(f"{BASE_DIRECTORY}/options.json")
if TESTING_FILE:
    options.testing_file = TESTING_FILE
else:
    options.testing_file = options.training_file.replace("training", TESTING_SOURCE)
options.num_dataloader_workers = 0

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

network = network.to_torchscript()

DATASET = network.validation_dataset

DATASET.return_index = True

dataloader_options = network.dataloader_options
dataloader_options["pin_memory"] = False
dataloader_options["num_workers"] = 0
dataloader_options["batch_size"] = 1
dataloader_options["drop_last"] = False
dataloader_options["shuffle"] = True

test_dataloader = network.dataloader(DATASET, **dataloader_options)

#scalene_profiler.start()

def process():
    for _, batch in zip(range(100), tqdm(test_dataloader)):
        (
            ids,
            batches1, 
            features1, 
            coordinates1, 
            targets1,
            object_targets1,
            batches2, 
            features2,
            coordinates2,
            targets2,
            object_targets2
        ) = batch

        # coordinates1 = torch.nn.functional.pad(coordinates1, (0,1), "constant", 0)
        # coordinates1[:,-1] = coordinates1[:,1]
        # coordinates2 = torch.nn.functional.pad(coordinates2, (0,1), "constant", 0)
        # coordinates2[:,-1] = coordinates2[:,1]
        
        predictions1, object_predictions1, predictions2, object_predictions2 = network.forward(
            features1, coordinates1, batches1, features2, coordinates2, batches2
        )

save_dir = Path("fil-result/pst")

with open(save_dir / "time.txt", "w") as text_file:
    pass

for i in range(51):
    start_time = time.perf_counter()
    profile(process, save_dir / f"{i}")
    t = (time.perf_counter() - start_time)
    print(t)
    with open(save_dir / "time.txt", "a") as text_file:
        text_file.write(str(t) + "\n")

# scalene_profiler.stop()
