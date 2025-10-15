# Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors

This repository contains code to train an HPST, and baselines like GAT and RCNN on NoVa Data for Multiple-view particle detector Segmentation


# Setup
We recommend using [conda](https://docs.conda.io/) for environment setup.  

```bash
git clone https://github.com/dikshantsagar/HPST-Nova.git
cd HPST-Nova
conda env create -f environment.yml
conda activate hpst
pip install --upgrade pip setuptools wheel
conda install -n base -c conda-forge mamba -y
mamba install -n hpst -c pytorch -c nvidia pytorch=2.5.1 pytorch-cuda=12.1 torchvision torchaudio -y

TORCH_VER=$(python -c "import torch as t; print(t.__version__.split('+')[0])")

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

echo "PYG URL: $PYG_URL"

pip install -f "$PYG_URL" torch_scatter torch_cluster torch_geometric
```

# Logging

We use WandB for logging. Please create a WandB project named "HPST" and use CLI login to use the code as is, or disable logging in the scripts.


# Training
Clone this repository.
```
git clone https://github.com/ayankele/dune-transformercvn.git
cd dune-transformercvn
```

To start the training, run 
```
python train.py -o "path of the option file" -n "create a name for your training"
```
For example, `python train.py -o option_files/fdhd_beam_2018prod_aiml_tutorial_2025_04_21.json -n tutorial_dense`

The option file is a required argument, and example option files are located in `option_files`. A description of each option is available in `transformercvn/options.py`. Among these are:  

`training_file`: Path to the training file  
`validation_file`: Optional path to the validation file  
`train_validation_split`: 0-1. If `validation_file` is not specified, this fraction of the dataset is randomly selected for training with the rest for validation.  
`num_gpu`: Number of available GPUs. 0 will use the CPU.  
`batch_size`: Number of events per batch at each training step.

Some options can be overridden through command-line arguments. Use `python train.py --help` for more information.

This repository supports different architectures for the CNNs responsible for embedding the pixel maps into the transformer. The default and the one we use in this tutorial is a DenseNet CNN.  
Add the argument `--sparse` to use a network with sparse convolutions. Note that this network cannot be exported to torchscript.   
Use `--sdxl` to use a Stable Diffusion XL inspired CNN with self-attention layers. This is the network we use in the LArSoft module.

The data we will use as specified in the example option file is in `/exp/dune/data/users/ayankele/fdhd_beam_2018prod_prong_pixels_minkowski_sparse.h5`

By default, a cosine learning rate schedule with warmup will be used and can be adjusted in the options file. The period of a cycle in epochs is `epochs` / `learning_rate_cycles`. Setting `learning_rate_cycles: 0` will use a decaying learning rate.

# Stopping and Resuming Training
Starting training will create a folder with the specified name and a subfolder `version_0`. Training can be stopped with Ctrl-C. Stopping at the end of an epoch is preferred since resuming will rerandomize the sequence of training events for the epoch. Subsequent trainings with the same name will increment the version number.
To resume training, add the argument `-c "path to model checkpoint"`.

# Testing
The jupyter notebook [Evaluate.ipynb](Evaluate.ipynb) can be used to both monitor training and to test a trained network.  
The "Training History" section will read the specified tensorboard created during training and plot several metrics as a function of training step. Metrics with the word `train` are saved each training step, and others are saved at the end of each validation run. There are separate metrics for event and prong classification performance. Metrics with `val` refer to the simple average of the corresponding event and prong metrics. The learning rate schedule can also be plotted here.

The "Testing" section will run the network against the specified dataset. For this tutorial, we will use the validation set, but a separate file conatining a testing set can be specified as `TESTING_FILE`.
This is followed by several testing metrics and plots such as ROC curves and confusion matrices.

# Compiling the network
The [CreateCompiled.ipynb](CreateCompiled.ipynb) jupyter notebook can be used to compile the network into a torchscript file for use in the C++ LArSoft ART framework. Modify the paths in the second cell to point to the directory and checkpoint of the model you want to export.
The final cell will create three torchscript files. All three models take as input a single tensor with the shape [(1+Npng), 3, 400, 280] where the first 3x400x280 image corresponds to the event pixel map followed by the prong pixel maps.   

The file ending in `pid` outputs a tuple of 2 tensors. The first tensor gives the softmax scores corresponding to the event classes. The second tensor is Npng x 8, corresponding to the prong softmax scores in the order that they were input.  

The file ending in `embeddings` also outputs a tuple of 2 tensors. Rather than the final event/prong predictions, this model outputs the intermediate feature representation vectors that are output by the transformer and serve as input into the classification layers. The first tensor has length 128 and corresponds to the event feature representation, and the second tensor is Npng x 128 and corresponds to the prong feature representations.  

The file ending in `combined` gives all of the above outputs as a tuple of 4 tensors: event prediction, prong predictions, event embedding, prong embeddings.  


## Train HPST
```bash
python scripts/train.py --options_file "config/hpst/hpst_tune_nova.json" --name "{run_name}" --log_dir "runs" --gpus 4 
```

## Train GAT
```bash
python scripts/train_gat.py --options_file "config/gnn/gat_tune_nova.json" --name "{run_name}" --log_dir "runs" --gpus 4
```

## Train RCNN
```bash
python scripts/train_rcnn.py --options_file "config/rcnn/rcnn_tune_nova.json" --name "{run_name}" --log_dir "runs" --gpus 8
```
