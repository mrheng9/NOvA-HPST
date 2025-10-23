# Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors
This tutorial based on the server @tau-neutrino.ps.uci.edu  

This repository contains code to train an HPST, and baselines like GAT and RCNN on NoVa Data for Multiple-view particle detector Segmentation


# Setup
We recommend using [conda](https://docs.conda.io/) for environment setup. 
You can run the command below to download Miniconda. 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
You can use my environment setted up already by running the command below.
```
conda activate /home/houyh/miniconda3/envs/hpst
```

Commands below are the whole process about setting up the environment
```bash
git clone https://github.com/mrheng9/NOvA-HPST.git
cd HPST-Nova
conda env create -f environment.yml
conda activate hpst
conda install -n base -c conda-forge mamba -y
mamba install -n hpst -c pytorch -c nvidia pytorch=2.5.1 pytorch-cuda=12.1 torchvision torchaudio -y

pip install --upgrade pip setuptools wheel
TORCH_VER=$(python -c "import torch as t; print(t.__version__.split('+')[0])")

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

echo "PYG URL: $PYG_URL"

pip install -f "$PYG_URL" torch_scatter torch_cluster torch_geometric

pip install -U lightning rich
```

# Logging

We use WandB for logging. For the first time training, please run the command below to attempt to train in your own terminal in order to create an account in WandB that stores real-time and complete training logs
```
python scripts/train.py --options_file "config/hpst/hpst_tune_nova.json" --name "hpst_run" --log_dir "runs" --gpus 1
```


# Training
To start the training, run 
```
HPST model:
bash train_hpst.sh

GAT model:
bash train_gat.sh

RCNN model:
bash train_rcnn.sh (not recommanded to run on tau server)
```
For example(commands in train_hpst.py), 
```
CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/train.py --options_file "config/hpst/hpst_tune_nova.json" --name "hpst run" --log_dir "runs" --gpus 2 > hpst.log 2>&1 &
```
The options_file is not a required argument. A description of each option is available in `hpst/utils/options.py`. 

The training log will be stored in the runs/hpst/wandb folder and the checkpoints will be stored under runs/hpst in the folder with the name of the specific moment you run it. 

We mainly foucus on the training and the performance of the HPST model and the results of GAT/RCNN will serve as the baseline to compare with the HPST's. However, the RCNN model is not recommanded to run on tau server for it has largest parameters.


The data we will use as specified in the example option file is in `/mnt/ironwolf_14t/users/ayankelevich/preprocessed_nova_miniprod6_1_cvnlabmaps.h5`

# Testing & Plottings
We provide a unified evaluator for HPST and GAT that reproduces all figures and metrics (confusion matrix, per-class accuracy, efficiency/purity, optional ROC curves).

To test the model, run
```
python scripts/evaluation.py --model gat/hpst --checkpoint_path "your checkpoint path" 
```   
## Notes
- `--model` and `checkpoin_path` are requiured when run the test
- use GPU 
By default the script uses CPU as defult. To force CPU, unset CUDA devices temporarily by adding: `--use_cuda False`

- Speed knobs  
--max_batches N to evaluate on a subset for a quick sanity check. **(Only first 100 batches are used in the tutorial)**    
--batch_size, --num_workers, --pin_memory to increase throughput.   
--examples_to_save 0 to skip event displays.  
--do_roc to additionally draw per-class ROC curves (AUC scores are always printed).


Outputs (saved under results/model)

- confusion_matrix.png
- confusion_matrix_normalized.png
- class_accuracy.png
- efficiency_purity_distribution.png
- roc_curves_per_class.png (only if -- do_roc is set)
- model_example_event*.png (event displays; count controlled by --examples_to_save)

Printed metrics

- Overall accuracy
- Per-class ROC-AUC scores (+ weighted average)
- Full classification report (precision/recall/F1)
- Per-class accuracy summary aligned with the confusion matrix

**You can find more visualization codes in HPST-Nova/hpst/notebooks**

# Compiling the network
This project provides a script to export a trained HPST checkpoint to a TorchScript module for C++ deployment (LArSoft/LibTorch).

What gets exported
- A single pointwise model (per-point logits), matching the training outputs.
- File name: hpst_{checkpoint-stem}.torchscript (saved next to the checkpoint by default).
- Export method: torch.jit.trace (no script).

Inputs and outputs
- Inputs (two views, same as training):
  - features1: [N1, 1]  // View X energy
  - coords1:   [N1, 2]  // View X coordinates
  - features2: [N2, 1]  // View Y energy
  - coords2:   [N2, 2]  // View Y coordinates
- Outputs (raw logits, not softmax):
  - event_logits1:  [N1, num_classes]
  - object_logits1: [N1, num_objects]
  - event_logits2:  [N2, num_classes]
  - object_logits2: [N2, num_objects]

How to run
- Ensure your checkpoint and options file are available. The options JSON must contain training_file so the script can reconstruct the dataset to fetch one example for tracing.
- From the repo root:
```
python CreateCompiled.py 
  --checkpoint "runs/hpst/<run>/checkpoints/last.ckpt" 
  --options "config/hpst/hpst_tune_nova.json" 
  --cuda 
```
- Arguments:
  - --checkpoint: path to .ckpt (required)
  - --options: path to options JSON (required)
  - --output-dir: output folder (default: checkpoint's folder)
  - --cuda (required)
  - --cuda-device: export on GPU (optional)

Notes and caveats
- The script uses torch.jit.trace. It runs one forward pass with a real sample from options.training_file (trainer.training_dataset[0]) to record the computation graph.
- The example sample is not saved in the .torchscript file; only the graph and weights are saved.
- Tracing assumes a fixed computation path (no data-dependent Python control flow). This repository’s export wrapper satisfies that.



