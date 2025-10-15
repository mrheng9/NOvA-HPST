from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import lightning.pytorch as pl
import torch
import sys
sys.path.append("./")
from pathlib import Path
import datetime
# torch.multiprocessing.set_sharing_strategy("file_system")
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    RichProgressBar,
    RichModelSummary
)

from hpst.utils.options import Options

@rank_zero_only
def update_config(logger, config_dict):
    logger.experiment.config.update(config_dict, allow_val_change=True)

def main(
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
        eval: int,
        **kwargs
):
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    master = "NODE_RANK" not in environ
    from hpst.trainers.masked_rcnn_trainer import MaskedRNNTrainer
    Network = MaskedRNNTrainer

    options = Options(training_file)
    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # Apply Command line overrides for common option values.
    # ---------------------------------------------------------------------------------------------
    options.verbose_output = verbose

    if threads is not None:
        if master:
            print(f"Setting CPU count: {threads}")

        torch.set_num_threads(threads)
        environ["OMP_NUM_THREADS"] = str(threads)
        environ["MKL_NUM_THREADS"] = str(threads)

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if debug:
        if master:
            print(f"Debug Mode: 1 GPU, 0 dataloader workers, Small Batch size")
        torch.jit.script = lambda x: x
        options.num_dataloader_workers = 0
        options.batch_size = 1

    # Print the full hyperparameter list
    # ---------------------------------------------------------------------------------------------
    if master:
        options.display()

    # Create the initial model on the CPU
    # ---------------------------------------------------------------------------------------------
    model = Network(options)

    # Create Loggers and Checkpoint systems
    # ---------------------------------------------------------------------------------------------
    if debug:
        logger = False
        callbacks = None

    else:
        # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
        #logger = TensorBoardLogger(save_dir="results/mask_rcnn", name=name, log_graph=graph)
        log_dir = getcwd() if log_dir is None else log_dir

        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        run_id = f"{timestamp}"
        base_dir = Path(log_dir) / "rcnn" / run_id  
        
        logger = WandbLogger(project='HPST', name=name, id='rcnn', save_dir=str(base_dir.parent))
        update_config(logger, options)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(base_dir / "checkpoints"),  
            filename="{epoch}-{step}-{val_accuracy:.4f}",
            verbose=options.verbose_output,
            every_n_train_steps=eval,
            monitor="val_accuracy",
            mode="max",
            save_top_k=5,
            save_last=True
        )

        callbacks = [
            checkpoint_callback,
            LearningRateMonitor(),
            TQDMProgressBar(refresh_rate=10), 
            #RichProgressBar(),
            RichModelSummary(max_depth=3)
        ]

    distributed_backend = None
    if options.num_gpu > 1:
        distributed_backend = DDPStrategy(
            find_unused_parameters=True
        )

    # Create the final pytorch-lightning manager
    # ---------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=options.epochs,
        callbacks=callbacks,
        #resume_from_checkpoint=checkpoint,
        strategy=distributed_backend,
        accelerator='gpu' if options.num_gpu > 0 else None,
        devices=options.num_gpu if options.num_gpu > 0 else None,
        #track_grad_norm=2 if options.verbose_output else -1,
        #gradient_clip_val=options.gradient_clip,
        #precision=16 if fp16 else 32,
        precision=16,
        val_check_interval=1.0,
        log_every_n_steps=3
    )

    if master and not debug:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.save_dir, exist_ok=True)
        with open(trainer.logger.save_dir + "/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-t", "--training_file", type=str, default="./data/preprocessed_nova_miniprod6_1_cvnlabmaps.h5",
                        help="Input file containing training data.")

    parser.add_argument("-o", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load from")

    parser.add_argument("-n", "--name", type=str, default="lightning_logs",
                        help="The sub-directory to create for this run.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs.")

    parser.add_argument("-fp16", action="store_true",
                        help="Use AMP for training.")

    parser.add_argument("-g", "--graph", action="store_true",
                        help="Log the computation graph.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-e", "--eval", type=int, default=500,
                        help="Number of steps before eval")

    parser.add_argument("--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")

    parser.add_argument("--threads", type=int, default=None,
                        help="Override CPU count in hyperparameters.")

    parser.add_argument("-p", "--pixels", action="store_true",
                        help="Use Pixel information in dataset.")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debug options super-switch. ")

    main(**parser.parse_args().__dict__)