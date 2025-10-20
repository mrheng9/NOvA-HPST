"""
Export trained models to TorchScript format for deployment
Supports HPST, GAT, and RCNN models
"""

import numpy as np
import torch
from torch import nn
from pathlib import Path
import json
import re
from glob import glob
from typing import Tuple, Optional

from hpst.utils.options import Options


class DynamicHPSTSimplified(nn.Module):
    """
    Simplified HPST model for inference
    Input: point cloud data
    Output: event and prong classifications
    """
    __constants__ = ["num_features", "num_classes", "num_objects"]
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        
        # Store dataset properties
        self.num_features = trainer.training_dataset.num_features
        self.num_classes = trainer.num_classes
        self.num_objects = trainer.num_objects
        
        # Store normalization parameters
        self.mean = trainer.mean
        self.std = trainer.std
        
    def forward(self, 
                features1: torch.Tensor,
                coords1: torch.Tensor,
                features2: torch.Tensor, 
                coords2: torch.Tensor):
        """
        Args:
            features1: [N_event, C] event point features
            coords1: [N_event, 3] event point coordinates
            features2: [N_prongs, C] prong features
            coords2: [N_prongs, 3] prong coordinates
        Returns:
            predictions1: [num_classes] event class probabilities
            predictions2: [num_classes] prong class probabilities
        """
        # Concatenate coordinates with features
        features1 = torch.cat([coords1, features1], dim=-1)
        features2 = torch.cat([coords2, features2], dim=-1)
        
        # Normalize features
        features1 = features1.clone()
        features1 = (features1 - self.mean) / self.std
        
        features2 = features2.clone()
        features2 = (features2 - self.mean) / self.std
        
        # Create dummy batch indices (single batch)
        batch1 = torch.zeros(features1.shape[0], dtype=torch.long, device=features1.device)
        batch2 = torch.zeros(features2.shape[0], dtype=torch.long, device=features2.device)
        
        # Run model
        outputs1, outputs2 = self.network((coords1, features1, batch1), (coords2, features2, batch2))
        
        # Split predictions and object predictions
        predictions1 = outputs1[:, :self.num_classes]
        predictions2 = outputs2[:, :self.num_classes]
        
        # Apply softmax
        predictions1 = torch.softmax(predictions1, dim=-1).mean(dim=0)  # Average over points
        predictions2 = torch.softmax(predictions2, dim=-1).mean(dim=0)
        
        return predictions1, predictions2


class DynamicHPSTEmbedding(nn.Module):
    """
    HPST model for extracting feature embeddings
    """
    __constants__ = ["num_features", "embed_channels"]
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        
        self.num_features = trainer.training_dataset.num_features
        # Get embed_channels from the network's first encoder
        self.embed_channels = trainer.num_classes + trainer.num_objects
        
        self.mean = trainer.mean
        self.std = trainer.std
        
    def forward(self,
                features1: torch.Tensor,
                coords1: torch.Tensor,
                features2: torch.Tensor,
                coords2: torch.Tensor):
        """
        Returns:
            event_embedding: [embed_channels] event feature vector
            prong_embedding: [embed_channels] prong feature vector
        """
        # Concatenate and normalize
        features1 = torch.cat([coords1, features1], dim=-1)
        features2 = torch.cat([coords2, features2], dim=-1)
        
        features1 = (features1.clone() - self.mean) / self.std
        features2 = (features2.clone() - self.mean) / self.std
        
        batch1 = torch.zeros(features1.shape[0], dtype=torch.long, device=features1.device)
        batch2 = torch.zeros(features2.shape[0], dtype=torch.long, device=features2.device)
        
        # Extract embeddings from encoder
        outputs1, outputs2 = self.network((coords1, features1, batch1), (coords2, features2, batch2))
        
        # Average pooling to get single embedding per set
        event_emb = outputs1.mean(dim=0)
        prong_emb = outputs2.mean(dim=0)
        
        return event_emb, prong_emb


class DynamicHPSTCombined(nn.Module):
    """
    Combined model: outputs both predictions and embeddings
    """
    __constants__ = ["num_features", "num_classes", "num_objects", "embed_channels"]
    
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.network
        
        self.num_features = trainer.training_dataset.num_features
        self.num_classes = trainer.num_classes
        self.num_objects = trainer.num_objects
        self.embed_channels = trainer.num_classes + trainer.num_objects
        
        self.mean = trainer.mean
        self.std = trainer.std
        
    def forward(self,
                features1: torch.Tensor,
                coords1: torch.Tensor,
                features2: torch.Tensor,
                coords2: torch.Tensor):
        """
        Returns:
            predictions1: [num_classes]
            predictions2: [num_classes]
            event_embedding: [embed_channels]
            prong_embedding: [embed_channels]
        """
        # Concatenate and normalize
        features1 = torch.cat([coords1, features1], dim=-1)
        features2 = torch.cat([coords2, features2], dim=-1)
        
        features1 = (features1.clone() - self.mean) / self.std
        features2 = (features2.clone() - self.mean) / self.std
        
        batch1 = torch.zeros(features1.shape[0], dtype=torch.long, device=features1.device)
        batch2 = torch.zeros(features2.shape[0], dtype=torch.long, device=features2.device)
        
        # Get outputs
        outputs1, outputs2 = self.network((coords1, features1, batch1), (coords2, features2, batch2))
        
        # Split predictions
        predictions1 = outputs1[:, :self.num_classes]
        predictions2 = outputs2[:, :self.num_classes]
        
        # Get embeddings (full output vectors)
        event_emb = outputs1.mean(dim=0)
        prong_emb = outputs2.mean(dim=0)
        
        # Apply softmax to predictions
        predictions1 = torch.softmax(predictions1, dim=-1).mean(dim=0)
        predictions2 = torch.softmax(predictions2, dim=-1).mean(dim=0)
        
        return predictions1, predictions2, event_emb, prong_emb


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint by step number"""
    checkpoints = list(checkpoint_dir.glob("epoch*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract step numbers
    steps = []
    for ckpt in checkpoints:
        match = re.search(r"step=(\d+)", ckpt.name)
        if match:
            steps.append(int(match.group(1)))
        else:
            steps.append(0)
    
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
    Export trained model to TorchScript
    
    Args:
        model_type: 'hpst', 'gat', or 'rcnn'
        checkpoint_path: Path to checkpoint, if None will find latest
        output_dir: Where to save outputs, defaults to checkpoint directory
        cuda: Whether to use GPU
        cuda_device: GPU device ID
    """
    
    # Import appropriate trainer
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
    
    # Load options
    print(f"Loading config from: {config_path}")
    options = Options.load(config_path)
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(f"results/{model_type}/checkpoints")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    print(f"Loading from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    
    # Create trainer and load weights
    print("Creating trainer...")
    trainer = Trainer(options)
    trainer.load_state_dict(state_dict)
    trainer.eval()
    
    # Freeze parameters
    for param in trainer.parameters():
        param.requires_grad_(False)
    
    if cuda:
        trainer = trainer.cuda(cuda_device)
    
    # Create export models
    print("\nCreating TorchScript models...")
    print("Note: TorchScript compilation may fail due to dynamic control flow.")
    print("Attempting to trace models instead...\n")
    
    # Get sample data for tracing
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        trainer.training_dataset, 
        batch_size=1, 
        collate_fn=trainer.dataloader_options["collate_fn"]
    )
    sample_batch = next(iter(dataloader))
    
    (_, batches1, features1, coordinates1, targets1, object_targets1, 
     batches2, features2, coordinates2, targets2, object_targets2) = sample_batch
    
    # Use tracing instead of scripting
    try:
        print("Tracing simplified model...")
        simplified_model = DynamicHPSTSimplified(trainer)
        simplified = torch.jit.trace(
            simplified_model,
            (features1, coordinates1, features2, coordinates2)
        )
        print("  ✓ Simplified model traced successfully")
    except Exception as e:
        print(f"  ✗ Failed to trace simplified model: {e}")
        import traceback
        traceback.print_exc()
        simplified = None
    
    try:
        print("Tracing embedding model...")
        embedding_model = DynamicHPSTEmbedding(trainer)
        embeddings = torch.jit.trace(
            embedding_model,
            (features1, coordinates1, features2, coordinates2)
        )
        print("  ✓ Embedding model traced successfully")
    except Exception as e:
        print(f"  ✗ Failed to trace embedding model: {e}")
        import traceback
        traceback.print_exc()
        embeddings = None
    
    try:
        print("Tracing combined model...")
        combined_model = DynamicHPSTCombined(trainer)
        combined = torch.jit.trace(
            combined_model,
            (features1, coordinates1, features2, coordinates2)
        )
        print("  ✓ Combined model traced successfully")
    except Exception as e:
        print(f"  ✗ Failed to trace combined model: {e}")
        import traceback
        traceback.print_exc()
        combined = None
    
    # Save models
    if output_dir is None:
        output_dir = checkpoint_path.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = checkpoint_path.stem
    saved_models = []
    
    if simplified is not None:
        simplified_path = output_dir / f"{model_type}_{basename}_simplified.torchscript"
        simplified.save(str(simplified_path))
        saved_models.append(("Simplified", simplified_path.name))
    
    if embeddings is not None:
        embeddings_path = output_dir / f"{model_type}_{basename}_embeddings.torchscript"
        embeddings.save(str(embeddings_path))
        saved_models.append(("Embeddings", embeddings_path.name))
    
    if combined is not None:
        combined_path = output_dir / f"{model_type}_{basename}_combined.torchscript"
        combined.save(str(combined_path))
        saved_models.append(("Combined", combined_path.name))
    
    if saved_models:
        print(f"\n✓ Export complete! Saved to: {output_dir}")
        for model_name, filename in saved_models:
            print(f"  - {model_name}: {filename}")
    else:
        print("\n✗ No models were successfully exported.")
        print("This is likely due to TorchScript limitations with dynamic graphs.")
        print("Consider using the model directly in Python instead.")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export models to TorchScript")
    parser.add_argument("--model", type=str, required=True, 
                       choices=['hpst', 'gat', 'rcnn'],
                       help="Model type to export")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (default: latest)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as checkpoint)")
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