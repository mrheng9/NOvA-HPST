# Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors

This repository contains code to train an HPST, and baselines like GAT and RCNN on NoVa Data for Multiple-view particle detector Segmentation


## Setup
We recommend using [conda](https://docs.conda.io/) for environment setup.  

```bash
git clone https://github.com/dikshantsagar/HPST-Nova.git
cd HPST-Nova
conda create -n hpst python=3.10
conda activate hpst
pip install -r requirements.txt
```

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
