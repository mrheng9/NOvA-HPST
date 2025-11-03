"""
Export trained HPST models to TorchScript format for deployment
"""

import torch
from torch import nn
from pathlib import Path
import json
import traceback
from argparse import ArgumentParser
from typing import Optional

from hpst.utils.options import Options
from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer


class HPSTPointwiseNetwork(nn.Module):
    """
    HPST model for C++ deployment - per-point predictions
    This matches the training output exactly
    """
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        self.num_classes = trainer.num_classes
        self.num_objects = trainer.num_objects
        self.register_buffer('mean', trainer.mean)
        self.register_buffer('std', trainer.std)
        
    def forward(self, features1, coords1, features2, coords2):
        """
        Args:
            features1: [N1, 1] - View X energy values
            coords1: [N1, 2] - View X coordinates
            features2: [N2, 1] - View Y energy values
            coords2: [N2, 2] - View Y coordinates
        
        Returns:
            particle_sofrmax1: [N1, num_classes=6] - Per-point particle classification
            object_softmax1: [N1, num_objects=10] - Per-point object ID
            particle_softmax2: [N2, num_classes=6] 
            object_softmax2: [N2, num_objects=10]
        """
        # Concatenate and normalize (same as training)
        full_features1 = torch.cat([coords1, features1], dim=-1)
        full_features2 = torch.cat([coords2, features2], dim=-1)
        full_features1 = (full_features1 - self.mean) / self.std
        full_features2 = (full_features2 - self.mean) / self.std
        
        # Create batch (single sample for inference)
        # batch1 = torch.zeros(coords1.shape[0], dtype=torch.long, device=coords1.device)
        # batch2 = torch.zeros(coords2.shape[0], dtype=torch.long, device=coords2.device)
        batch1 = coords1.new_zeros((coords1.shape[0],), dtype=torch.long)
        batch2 = coords2.new_zeros((coords2.shape[0],), dtype=torch.long)
        # Forward pass
        outputs1, outputs2 = self.network(
            (coords1, full_features1, batch1),
            (coords2, full_features2, batch2)
        )
        
        # Split outputs (same as training)
        particle_softmax1 = outputs1[:, :self.num_classes]      # [N1, 6]
        object_softmax1 = outputs1[:, self.num_classes:]     # [N1, 10]
        particle_softmax2 = outputs2[:, :self.num_classes]      # [N2, 6]
        object_softmax2 = outputs2[:, self.num_classes:]     # [N2, 10]
        
        return particle_softmax1, object_softmax1, particle_softmax2, object_softmax2

class HPSTPointwiseSoftmax(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
    def forward(self, features1, coords1, features2, coords2):
        e1, o1, e2, o2 = self.core(features1, coords1, features2, coords2)
        return (
            torch.softmax(e1, dim=-1),
            torch.softmax(o1, dim=-1),
            torch.softmax(e2, dim=-1),
            torch.softmax(o2, dim=-1),
        )

def load_trainer_from_checkpoint(checkpoint_path: Path, options_file: Optional[Path] = None):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu',weights_only=False)
    
    # Load options
    if options_file is not None and options_file.exists():
        with open(options_file, 'r') as f:
            options_dict = json.load(f)
        options = Options(options_dict['training_file'])
        options.update_options(options_dict)
    elif 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        if 'options' in hparams:
            options = hparams['options']
        else:
            options = Options(hparams['training_file'])
            for key, value in hparams.items():
                if hasattr(options, key):
                    setattr(options, key, value)
    else:
        raise ValueError("Cannot load options")
    
    # Load trainer
    trainer = HeterogenousPointSetTrainer.load_from_checkpoint(
        str(checkpoint_path), options=options, map_location='cpu'
    )
    trainer.eval()
    for param in trainer.parameters():
        param.requires_grad_(False)
    
    print(f"✓ Loaded trainer (classes={trainer.num_classes}, objects={trainer.num_objects})")
    return trainer


def get_example_data(trainer, device):
    print("\nLoading example data...")
    dataset = trainer.training_dataset
    hits_index, view_x, view_y = dataset[0]
    
    features_x, coords_x, _, _ = view_x
    features_y, coords_y, _, _ = view_y
    
    features1 = features_x.to(device)
    coords1 = coords_x.to(device)
    features2 = features_y.to(device)
    coords2 = coords_y.to(device)
    
    if coords1.dim() == 3:
        coords1, features1 = coords1.squeeze(0), features1.squeeze(0)
    if coords2.dim() == 3:
        coords2, features2 = coords2.squeeze(0), features2.squeeze(0)
    
    print(f"✓ Example data: view1={coords1.shape[0]} pts, view2={coords2.shape[0]} pts")
    return features1, coords1, features2, coords2


def export_model(checkpoint_path, options_file=None, output_dir=None, cuda=False, cuda_device=0, probs=False):
     # Load
    trainer = load_trainer_from_checkpoint(checkpoint_path, options_file)
    
    # Device
    device = torch.device(f'cuda:{cuda_device}' if cuda else 'cpu')
    if cuda:
        trainer = trainer.to(device)
    
    # Example data
    features1, coords1, features2, coords2 = get_example_data(trainer, device)
    
    # Export single pointwise model
    print("\n" + "="*60)
    print("Exporting HPST Pointwise Model (trace)")
    print("="*60)
    
    # model = HPSTPointwiseNetwork(trainer).to(device).eval()
    
    # with torch.no_grad():
    #     traced = torch.jit.trace(
    #         model,
    #         (features1, coords1, features2, coords2),
    #         check_trace=False
    #     )
    #     # Test
    #     output = traced(features1, coords1, features2, coords2)

    model = HPSTPointwiseNetwork(trainer).eval()
    if probs:
        model = HPSTPointwiseSoftmax(model).eval()
    # Always trace on CPU to avoid baking CUDA into the graph
    model = model.to('cpu')
    features1_cpu = features1.to('cpu')
    coords1_cpu = coords1.to('cpu')
    features2_cpu = features2.to('cpu')
    coords2_cpu = coords2.to('cpu')

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (features1_cpu, coords1_cpu, features2_cpu, coords2_cpu),
            check_trace=False
        )
        # Test on CPU
        output = traced(features1_cpu, coords1_cpu, features2_cpu, coords2_cpu)


        print(f"✓ Output shapes:")
        print(f"  Particle softmax view1: {output[0].shape}")  # [N1, 6]
        print(f"  Object softmax view1: {output[1].shape}") # [N1, 10]
        print(f"  Particle softmax view2: {output[2].shape}")  # [N2, 6]
        print(f"  Object softmax view2: {output[3].shape}") # [N2, 10]
    
    # Save
    if output_dir is None:
        output_dir = checkpoint_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = checkpoint_path.stem
    model_path = output_dir / f"hpst_{basename}.torchscript"
    traced.save(str(model_path))
    
    print(f"\n{'='*60}")
    print(f"✓ Export Complete!")
    print(f"  Saved: {model_path.name}")
    print(f"{'='*60}")
    
    return output_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="HPST/jdyrag4x/checkpoints/last.ckpt")
    parser.add_argument("--options", default="/home/houyh/HPST-Nova/config/hpst/hpst_tune_nova.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--probs", default=True, help="Export softmax probabilities instead of softmax")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    options_file = Path(args.options) if args.options else None
    if options_file and not options_file.exists():
        options_file = None
    
    if options_file is None:
        potential = checkpoint_path.parent.parent / "options.json"
        if potential.exists():
            options_file = potential
    
    export_model(
        checkpoint_path=checkpoint_path,
        options_file=options_file,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cuda=args.cuda,
        cuda_device=args.cuda_device,
        probs=args.probs
    )