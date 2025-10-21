"""
Export trained HPST models to TorchScript format for deployment
"""

import numpy as np
import torch
from torch import nn
from pathlib import Path
import re
from typing import Tuple, Optional
import traceback

from hpst.utils.options import Options


class DynamicHPSTSimplified(nn.Module):
    """
    Simplified HPST model for inference
    Outputs aggregated predictions per view
    """
    __constants__ = ["num_features", "num_classes", "num_objects"]
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        
        self.num_features = trainer.training_dataset.num_features
        self.num_classes = trainer.num_classes
        self.num_objects = trainer.num_objects
        
        self.mean = trainer.mean
        self.std = trainer.std
        
    def forward(self, 
                features1: torch.Tensor,
                coords1: torch.Tensor,
                features2: torch.Tensor, 
                coords2: torch.Tensor):
        """
        Args:
            features1: [N_view1, C] view1 point features
            coords1: [N_view1, 2] view1 point coordinates (X, YZ)
            features2: [N_view2, C] view2 point features
            coords2: [N_view2, 2] view2 point coordinates (X, YZ)
        
        Returns:
            event_pred1: [num_classes] view1 event class probabilities
            event_pred2: [num_classes] view2 event class probabilities
            object_pred1: [num_objects] view1 object probabilities
            object_pred2: [num_objects] view2 object probabilities
        """
        # Concatenate coordinates with features
        features1 = torch.cat([coords1, features1], dim=-1)
        features2 = torch.cat([coords2, features2], dim=-1)
        
        # Normalize (consistent with training)
        features1 = (features1 - self.mean) / self.std
        features2 = (features2 - self.mean) / self.std
        
        # Single batch (one event)
        batch1 = torch.zeros(features1.shape[0], dtype=torch.long, device=features1.device)
        batch2 = torch.zeros(features2.shape[0], dtype=torch.long, device=features2.device)
        
        # Forward pass
        outputs1, outputs2 = self.network((coords1, features1, batch1), 
                                          (coords2, features2, batch2))
        
        # Split outputs
        event_logits1 = outputs1[:, :self.num_classes]
        object_logits1 = outputs1[:, self.num_classes:]
        event_logits2 = outputs2[:, :self.num_classes]
        object_logits2 = outputs2[:, self.num_classes:]
        
        # Aggregate: mean of logits then softmax
        event_pred1 = torch.softmax(event_logits1.mean(dim=0), dim=-1)
        event_pred2 = torch.softmax(event_logits2.mean(dim=0), dim=-1)
        object_pred1 = torch.softmax(object_logits1.mean(dim=0), dim=-1)
        object_pred2 = torch.softmax(object_logits2.mean(dim=0), dim=-1)
        
        return event_pred1, event_pred2, object_pred1, object_pred2


class DynamicHPSTPointwise(nn.Module):
    """
    HPST model outputting per-point predictions
    """
    __constants__ = ["num_features", "num_classes", "num_objects"]
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        
        self.num_features = trainer.training_dataset.num_features
        self.num_classes = trainer.num_classes
        self.num_objects = trainer.num_objects
        
        self.mean = trainer.mean
        self.std = trainer.std
        
    def forward(self,
                features1: torch.Tensor,
                coords1: torch.Tensor,
                features2: torch.Tensor,
                coords2: torch.Tensor):
        """
        Args:
            features1: [N_view1, C]
            coords1: [N_view1, 2]
            features2: [N_view2, C]
            coords2: [N_view2, 2]
        
        Returns:
            event_preds1: [N_view1, num_classes]
            event_preds2: [N_view2, num_classes]
            object_preds1: [N_view1, num_objects]
            object_preds2: [N_view2, num_objects]
        """
        features1 = torch.cat([coords1, features1], dim=-1)
        features2 = torch.cat([coords2, features2], dim=-1)
        
        features1 = (features1 - self.mean) / self.std
        features2 = (features2 - self.mean) / self.std
        
        batch1 = torch.zeros(features1.shape[0], dtype=torch.long, device=features1.device)
        batch2 = torch.zeros(features2.shape[0], dtype=torch.long, device=features2.device)
        
        outputs1, outputs2 = self.network((coords1, features1, batch1),
                                          (coords2, features2, batch2))
        
        event_preds1 = torch.softmax(outputs1[:, :self.num_classes], dim=-1)
        object_preds1 = torch.softmax(outputs1[:, self.num_classes:], dim=-1)
        event_preds2 = torch.softmax(outputs2[:, :self.num_classes], dim=-1)
        object_preds2 = torch.softmax(outputs2[:, self.num_classes:], dim=-1)
        
        return event_preds1, event_preds2, object_preds1, object_preds2


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint by step number"""
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    steps = []
    for ckpt in checkpoints:
        match = re.search(r"step[=_](\d+)", ckpt.name)
        if match:
            steps.append(int(match.group(1)))
        else:
            steps.append(ckpt.stat().st_mtime)
    
    latest_idx = np.argmax(steps)
    return checkpoints[latest_idx]


def export_model(
    model_type: str,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cuda: bool = False,
    cuda_device: int = 0
):
    """
    Export trained model to TorchScript using torch.jit.script
    
    Args:
        model_type: 'hpst', 'gat', or 'rcnn'
        checkpoint_path: Path to checkpoint
        output_dir: Where to save outputs
        cuda: Whether to use GPU
        cuda_device: GPU device ID
    """
    
    # Import trainer
    if model_type == 'hpst':
        from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer as Trainer
        config_path = Path("config/hpst/hpst_tune_nova.json")
    elif model_type == 'gat':
        from hpst.trainers.gat_trainer import GATTrainer as Trainer
        config_path = Path("config/gat/gat_tune_nova.json")
    elif model_type == 'rcnn':
        from hpst.trainers.masked_rcnn_trainer import MaskedRCNNTrainer as Trainer
        config_path = Path("config/rcnn/rcnn_tune_nova.json")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Loading config from: {config_path}")
    options = Options.load(config_path)
    
    if checkpoint_path is None:
        checkpoint_dir = Path(f"results/{model_type}/checkpoints")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    
    print("Creating trainer...")
    trainer = Trainer(options)
    trainer.load_state_dict(state_dict)
    trainer.eval()
    
    for param in trainer.parameters():
        param.requires_grad_(False)
    
    device = torch.device(f'cuda:{cuda_device}' if cuda else 'cpu')
    if cuda:
        print(f"Moving model to GPU {cuda_device}")
        trainer = trainer.to(device)
    
    print(f"\nDevice: {device}")
    print("Export method: torch.jit.script\n")
    
    # Export models
    saved_models = []
    
    # 1. Simplified model
    try:
        print("="*60)
        print("Exporting Simplified Model")
        print("="*60)
        
        simplified_model = DynamicHPSTSimplified(trainer)
        simplified = torch.jit.script(simplified_model)
        
        saved_models.append(('simplified', simplified))
        print("✓ Simplified model scripted successfully")
        
    except Exception as e:
        print(f"✗ Failed to script simplified model: {e}")
        traceback.print_exc()
    
    # 2. Pointwise model
    try:
        print("\n" + "="*60)
        print("Exporting Pointwise Model")
        print("="*60)
        
        pointwise_model = DynamicHPSTPointwise(trainer)
        pointwise = torch.jit.script(pointwise_model)
        
        saved_models.append(('pointwise', pointwise))
        print("✓ Pointwise model scripted successfully")
        
    except Exception as e:
        print(f"✗ Failed to script pointwise model: {e}")
        traceback.print_exc()
    
    # Save models
    if output_dir is None:
        output_dir = checkpoint_path.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = checkpoint_path.stem
    
    print(f"\n{'='*60}")
    print(f"Saving to: {output_dir}")
    print(f"{'='*60}")
    
    for model_name, model in saved_models:
        model_path = output_dir / f"{model_type}_{basename}_{model_name}_scripted.pt"
        model.save(str(model_path))
        print(f"✓ {model_name}: {model_path.name}")
    
    if saved_models:
        print(f"\n{'='*60}")
        print("✓ Export Complete!")
        print(f"{'='*60}")
    else:
        print(f"\n✗ No models were successfully exported")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export HPST models to TorchScript")
    parser.add_argument("--model", type=str, required=True,
                       choices=['hpst', 'gat', 'rcnn'],
                       help="Model type to export")
    parser.add_argument("--checkpoint", type=str, 
                       default="/home/houyh/HPST-Nova/HPST/jdyrag4x/checkpoints/last.ckpt",
                       help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA")
    parser.add_argument("--cuda-device", type=int, default=0,
                       help="CUDA device ID")
    
    args = parser.parse_args()
    
    export_model(
        model_type=args.model,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cuda=args.cuda,
        cuda_device=args.cuda_device
    )