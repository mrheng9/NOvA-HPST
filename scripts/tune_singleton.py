from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import lightning.pytorch as pl
import torch

# torch.multiprocessing.set_sharing_strategy("file_system")

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

import torch

from hpst.utils.options import Options

"""
log_dir: str,
        name: str,
        options_file: str,
        training_file: str,
        checkpoint: Optional[str],
        fp16: bool,
        graph: bool,
        verbose: bool,
        batch_size: Optional[int],
        gpus: Optional[int],
        threads: Optional[int],
        debug: bool,
        network_type: str,
        eval: int,
        **kwargs
"""

def main(config):
    master = "NODE_RANK" not in environ

    
    # from dune.network.trainers.neutrino_point_set_pixel_trainer import NeutrinoPointSetPixelTrainer
    # from dune.network.trainers.heterogenous_neutrino_point_set_pixel_trainer import NeutrinoPointSetPixelTrainer
    # from dune.network.trainers.independent_heterogenous_neutrino_point_set_trainer_v3 import NeutrinoPointSetPixelTrainer
    # from dune.network.trainers.layered_heterogenous_neutrino_point_set_trainer_v3 import NeutrinoPointSetPixelTrainer
    from hpst.trainers.singleton_point_set_trainer import PointSetTrainer
    Network = PointSetTrainer

    options = Options("")

    if config["options_file"] is not None:
        with open(config["options_file"], 'r') as json_file:
            options.update_options(json.load(json_file))

    num_stages = config["num_stages"]
    options.update_options({
        "pointnet_enc_depths": [2,2,4,2][-num_stages:],
        "pointnet_enc_channels": [64, 128, 256, 512][-num_stages:],
        "pointnet_enc_groups": [12, 24, 48, 64][-num_stages:],
        "pointnet_enc_neighbours": [config["neighbors"], config["neighbors"], config["neighbors"], config["neighbors"]][-num_stages:],
        "pointnet_dec_neighbours": [config["neighbors"], config["neighbors"], config["neighbors"], config["neighbors"]][-num_stages:],
        "pointnet_dec_depths": [1, 1, 1, 1][-num_stages:],
        "pointnet_dec_channels": [32, 64, 128, 256][-num_stages:],
        "pointnet_dec_groups": [4, 8, 16, 32][-num_stages:],
        "pointnet_dec_neighbours": [config["neighbors"], config["neighbors"], config["neighbors"], config["neighbors"]][-num_stages:],
        "pointnet_grid_sizes": [config["base_grid_size"], config["base_grid_size"]*2, config["base_grid_size"]*4, config["base_grid_size"]*16][-num_stages:],
        "learning_rate": config["learning_rate"],
        "pointnet_patch_embed_channels": config["pointnet_patch_embed_channels"],
        "pointnet_patch_embed_neighbours": config["neighbors"],
        "epochs": options.epochs // options.learning_rate_cycles, #config["epochs"]//config["learning_rate_cycles"],
        "learning_rate_cycles": 1, 
        "num_dataloader_workers": 4
    })

    # Print the full hyperparameter list
    # ---------------------------------------------------------------------------------------------
    if master:
        options.display()

    # Create the initial model on the CPU
    # ---------------------------------------------------------------------------------------------
    model = Network(options, train_perc=0.1)

    # Create the final pytorch-lightning manager
    # ---------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=options.epochs,
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        strategy=RayDDPStrategy(find_unused_parameters=True),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=options.gradient_clip,
        val_check_interval=config["eval"],
        enable_progress_bar=False
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-o", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")
    
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    options_file = parser.parse_args().__dict__["options_file"]

    search_space = {
        "options_file": tune.choice([options_file]),
        "eval": tune.choice([1.0]),
        "num_stages": tune.choice([2,3,4]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "pointnet_patch_embed_channels": tune.choice([128, 256, 512]),
        "neighbors": tune.choice([4,8]),
        "base_grid_size": tune.choice([4, 8, 16])
    }

    options = Options("")

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # The maximum training epochs
    num_epochs = options.epochs // options.learning_rate_cycles

    # Number of samples from parameter space
    num_samples = 60

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    scaling_config = ScalingConfig(
        num_workers=4, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path="/home/roblesee/dune/hpst/ray_results",
        name="hpst_tune_singleton",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    from ray.train.torch import TorchTrainer
    #from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.search.basic_variant import BasicVariantGenerator

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        main,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    #bayesopt = BayesOptSearch(metric="ptl/val_accuracy", mode="max")
    bvg = BasicVariantGenerator(random_state=42)
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=bvg,
        ),
    )

    results = tuner.fit()
