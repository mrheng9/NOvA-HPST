from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import lightning.pytorch as pl
import torch
import sys
sys.path.append('./')
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
    from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer
    Network = HeterogenousPointSetTrainer 

    options = Options("")
    if config["options_file"] is not None:
        with open(config["options_file"], 'r') as json_file:
            options.update_options(json.load(json_file))

    num_stages = config["num_stages"]
    options.update_options({
        "pointnet_enc_depths": [2,2,6,2][-num_stages:],
        "pointnet_enc_channels": [96, 192, 384, 512][-num_stages:],
        "pointnet_enc_groups": [12, 24, 48, 64][-num_stages:],
        "pointnet_enc_neighbours": [config["neighbors"], config["neighbors"], config["neighbors"], config["neighbors"]][-num_stages:],
        "pointnet_dec_neighbours": [config["neighbors"], config["neighbors"], config["neighbors"], config["neighbors"]][-num_stages:],
        "pointnet_dec_depths": [1, 1, 1, 1][-num_stages:],
        "pointnet_dec_channels": [48, 96, 192, 384][-num_stages:],
        "pointnet_dec_groups": [6, 12, 24, 48][-num_stages:],
        "pointnet_dec_neighbours": [16, 16, 16, 16][-num_stages:],
        "pointnet_grid_sizes": [8, 16, 32, 64][-num_stages:],
        "learning_rate": config["learning_rate"],
        "pointnet_patch_embed_channels": config["pointnet_patch_embed_channels"],
        "pointnet_patch_embed_neighbours": config["neighbors"],
        "epochs": 50,
        "learning_rate_cycles": 1, 
        "num_dataloader_workers": 4
    })

    # Print the full hyperparameter list
    # ---------------------------------------------------------------------------------------------
    if master:
        options.display()

    # Create the initial model on the CPU
    # ---------------------------------------------------------------------------------------------
    model = Network(options, train_perc=0.01)

    # Create the final pytorch-lightning manager
    # ---------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=options.epochs,
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        strategy=RayDDPStrategy(),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=options.gradient_clip,
        val_check_interval=1.0,#config["eval"],
        log_every_n_steps=3,
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

    search_space = {
        "options_file": tune.choice([parser.parse_args().__dict__["options_file"]]),
        # "eval": tune.choice([68]),
        "num_stages": tune.choice([2,3,4]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "pointnet_patch_embed_channels": tune.choice([128, 256, 512]),
        "neighbors": tune.choice([4,8])
    }

    # The maximum training iterations
    num_iters = 50

    # Number of samples from parameter space
    num_samples = 60

    scheduler = ASHAScheduler(max_t=num_iters, grace_period=5, reduction_factor=2)

    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    scaling_config = ScalingConfig(
        num_workers=4, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path="/baldig/physicsprojects2/dikshans/ray_results",
        name="hpst_tune_small",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
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
    print(results.get_best_result(metric="val_accuracy", mode="max").config)
