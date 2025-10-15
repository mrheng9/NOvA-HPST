import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from hpst.utils.options import Options
from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer

# ========= Configuration =========
USE_CUDA = True  # Set True for GPU, False for CPU
CUDA_DEVICE_INDEX = 1
# Speed knobs
MAX_BATCHES = None      
EXAMPLES_TO_SAVE = 3 
DO_ROC = False

# File paths - UPDATE THESE
OPTIONS_PATH = "/home/houyh/HPST-Nova/config/hpst/hpst_tune_nova.json"
CHECKPOINT_PATH = "/home/houyh/HPST-Nova/HPST/jdyrag4x/checkpoints/last.ckpt"
TESTING_FILE = "/mnt/ironwolf_14t/users/ayankelevich/preprocessed_nova_miniprod6_1_cvnlabmaps.h5"

# ========= Device Setup =========
if USE_CUDA and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_INDEX)
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
    USE_CUDA = False

def to_device(x):
    return x.to(DEVICE, non_blocking=True) if torch.is_tensor(x) else x

def _to_1d(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    return x.reshape(-1)

def safe_sample_id(ids_obj, local_idx: int, batch_idx: int | str):
    try:
        if torch.is_tensor(ids_obj):
            if ids_obj.dim() == 0:
                return int(ids_obj.item())
            flat = ids_obj.reshape(-1)
            return int(flat[min(local_idx, flat.numel() - 1)].item())
        if isinstance(ids_obj, (list, tuple, np.ndarray)):
            if len(ids_obj) > 0:
                idx = min(local_idx, len(ids_obj) - 1)
                val = ids_obj[idx]
                if torch.is_tensor(val):
                    return int(val.item())
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                    vv = val[0]
                    return int(vv.item() if torch.is_tensor(vv) else int(vv))
                return int(val)
    except Exception:
        pass
    return f"{batch_idx}_{local_idx}"

# ========== NOvA Dataset Configuration (6 classes) ==========
# Based on the paper's Figure 4(b) legend
NOVA_CLASS_NAMES = [
    "Background",      # Class 0 
    "Muon",           # Class 1
    "Electron",       # Class 2 
    "Proton",         # Class 3 
    "Photon",         # Class 4 
    "Pion"            # Class 5 
]

# Colors matching the paper's Figure 4(b) legend
NOVA_PALETTE = [
    "#F5F5F5",  # Background (light grey/white)
    "#1E88E5",  # Muon (blue)
    "#FF8C00",  # Electron (orange)
    "#43A047",  # Proton (green)
    "#E53935",  # Photon (red)
    "#8E24AA"   # Pion (purple)
]

NUM_CLASSES = len(NOVA_CLASS_NAMES)

def get_current_class_names(dataset, num_classes: int):
    """Get class names from dataset"""
    # Try to get class names from dataset attributes
    for src in (getattr(dataset, "class_names", None), getattr(dataset, "classes", None)):
        if isinstance(src, (list, tuple)) and len(src) == num_classes:
            return list(src)
    # Return NOvA names as default
    return NOVA_CLASS_NAMES if num_classes == 6 else [f"Class {i}" for i in range(num_classes)]


def build_label_remap(current_names, target_names=NOVA_CLASS_NAMES):
    """Build remapping from current class names to NOvA class names"""
    cur2idx = {str(n).lower().strip(): i for i, n in enumerate(current_names)}
    tgt2idx = {str(n).lower().strip(): i for i, n in enumerate(target_names)}
    remap = torch.arange(len(current_names), dtype=torch.long)
    
    # Aliases for common variations
    aliases = {
        # Background
        "bg": "background", "bkg": "background", "none": "background",
        
        # Muon
        "mu": "muon", "mu-": "muon", "muon-": "muon",
        
        # Electron
        "e": "electron", "e-": "electron", "electron-": "electron", 
        "electron shower": "electron", "em shower": "electron",
        
        # Proton
        "p": "proton", "p+": "proton", "proton+": "proton",
        
        # Photon
        "gamma": "photon", "γ": "photon", "photon shower": "photon",
        
        # Pion
        "pi": "pion", "π": "pion", "pi+": "pion", "pi-": "pion",
        "pion+": "pion", "pion-": "pion", "charged pion": "pion",
        "neutral pion": "pion", "pi0": "pion"
    }
    
    # Try to map each current class to target class
    for cur_i, cur_n in enumerate(current_names):
        key = str(cur_n).lower().strip()
        
        # Direct match
        j = tgt2idx.get(key, None)
        
        # Try aliases
        if j is None and key in aliases:
            j = tgt2idx.get(aliases[key], None)
        
        # If still not found, keep original index
        if j is None:
            j = cur_i if cur_i < len(target_names) else 0  # Default to background
        
        remap[cur_i] = j
    
    return remap

def apply_remap(labels_tensor: torch.Tensor, remap: torch.Tensor):
    """Apply label remapping"""
    if labels_tensor.dtype != torch.long:
        labels_tensor = labels_tensor.long()
    out = labels_tensor.clone()
    m = (out >= 0) & (out < remap.numel())
    out[m] = remap[out[m]]
    return out

def segment_majority_labels(cluster_ids: torch.Tensor | None, point_logits: torch.Tensor):
    """Apply majority voting within each cluster/object"""
    point_pred = torch.argmax(point_logits, dim=1).cpu()
    if cluster_ids is None:
        return point_pred
    
    cid = cluster_ids.cpu().view(-1)
    out = point_pred.clone()
    num_classes = point_logits.shape[1]
    
    for oid in torch.unique(cid):
        if oid.item() < 0:
            continue
        m = (cid == oid)
        if m.any():
            counts = torch.bincount(point_pred[m], minlength=num_classes)
            out[m] = int(torch.argmax(counts))
    return out

def extract_xz(coords: torch.Tensor):
    """Extract X-Z coordinates"""
    c = coords.cpu().numpy()
    if c.shape[1] >= 3:
        return c[:, 2], c[:, 0]  # Z, X
    return c[:, 0], c[:, 1]

def extract_yz(coords: torch.Tensor):
    """Extract Y-Z coordinates"""
    c = coords.cpu().numpy()
    if c.shape[1] >= 3:
        return c[:, 2], c[:, 1]  # Z, Y
    return c[:, 0], c[:, 1]

def plot_event(sample_id, coords1, y_true1, y_pred1, coords2, y_true2, y_pred2,
               class_names, palette, save_path):
    """Plot event visualization similar to Figure 4 with square markers"""
    colors = np.array(palette)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Marker style: square blocks
    marker_style = 's'  # square
    marker_size = 15    # larger size
    marker_edge = 0.5   # thin edge
    
    # XZ true label
    zx_z, zx_x = extract_xz(coords1)
    axes[0, 0].scatter(zx_z, zx_x, c=colors[y_true1.numpy()], 
                      marker=marker_style, s=marker_size, 
                      linewidths=marker_edge, edgecolors='none')
    axes[0, 0].set_title(f"XZ, true label, sample {sample_id}", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Z", fontsize=10)
    axes[0, 0].set_ylabel("X", fontsize=10)
    axes[0, 0].grid(True, alpha=0.2, linestyle='--')
    axes[0, 0].set_facecolor('#f8f8f8')

    # YZ true label
    zy_z, zy_y = extract_yz(coords2)
    axes[0, 1].scatter(zy_z, zy_y, c=colors[y_true2.numpy()], 
                      marker=marker_style, s=marker_size, 
                      linewidths=marker_edge, edgecolors='none')
    axes[0, 1].set_title(f"YZ, true label, sample {sample_id}", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Z", fontsize=10)
    axes[0, 1].set_ylabel("Y", fontsize=10)
    axes[0, 1].grid(True, alpha=0.2, linestyle='--')
    axes[0, 1].set_facecolor('#f8f8f8')

    # XZ prediction
    axes[1, 0].scatter(zx_z, zx_x, c=colors[y_pred1.numpy()], 
                      marker=marker_style, s=marker_size, 
                      linewidths=marker_edge, edgecolors='none')
    axes[1, 0].set_title(f"XZ, prediction, sample {sample_id}", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Z", fontsize=10)
    axes[1, 0].set_ylabel("X", fontsize=10)
    axes[1, 0].grid(True, alpha=0.2, linestyle='--')
    axes[1, 0].set_facecolor('#f8f8f8')

    # YZ prediction
    axes[1, 1].scatter(zy_z, zy_y, c=colors[y_pred2.numpy()], 
                      marker=marker_style, s=marker_size, 
                      linewidths=marker_edge, edgecolors='none')
    axes[1, 1].set_title(f"YZ, prediction, sample {sample_id}", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Z", fontsize=10)
    axes[1, 1].set_ylabel("Y", fontsize=10)
    axes[1, 1].grid(True, alpha=0.2, linestyle='--')
    axes[1, 1].set_facecolor('#f8f8f8')

    # Legend with square markers
    handles = [Line2D([0], [0], marker='s', color='w', label=class_names[i],
                      markerfacecolor=palette[i], markersize=10, 
                      markeredgewidth=0.5, markeredgecolor='black')
               for i in range(len(class_names))]
    
    # Place legend outside the plot
    legend = axes[0, 1].legend(handles=handles, 
                               bbox_to_anchor=(1.05, 1.0), 
                               loc='upper left', 
                               borderaxespad=0., 
                               fontsize=10,
                               frameon=True,
                               fancybox=True,
                               shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # Add overall title
    fig.suptitle(f'Event Display - Sample {sample_id}', 
                fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.99])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_efficiency_purity(predictions, targets, class_names, palette, save_dir):
    """Plot efficiency and purity distributions per class (Figure 3 style)"""
    n_classes = len(class_names)
    
    # Calculate efficiency and purity for each sample
    efficiencies = {i: [] for i in range(n_classes)}
    purities = {i: [] for i in range(n_classes)}
    
    for true_class in range(n_classes):
        mask_true = (targets == true_class)
        if mask_true.sum() == 0:
            continue
            
        for pred_class in range(n_classes):
            mask_pred = (predictions == pred_class)
            
            # True positives
            tp = (mask_true & mask_pred).sum()
            
            # Efficiency: TP / (TP + FN)
            efficiency = tp / mask_true.sum() if mask_true.sum() > 0 else 0
            efficiencies[true_class].append(float(efficiency))
            
            # Purity: TP / (TP + FP)
            purity = tp / mask_pred.sum() if mask_pred.sum() > 0 else 0
            purities[true_class].append(float(purity))
    
    # Plot
    fig, axes = plt.subplots(2, n_classes, figsize=(20, 8))
    
    for i in range(n_classes):
        # Efficiency distribution
        if len(efficiencies[i]) > 0:
            axes[0, i].hist(efficiencies[i], bins=50, color=palette[i], alpha=0.7)
            mean_eff = np.mean(efficiencies[i])
            axes[0, i].axvline(mean_eff, color='red', linestyle='--', 
                              label=f'mean eff={mean_eff:.2f}')
            axes[0, i].set_title(f"{class_names[i]}")
            axes[0, i].set_xlabel("Efficiency")
            axes[0, i].set_ylabel("Count")
            axes[0, i].legend(fontsize=8)
            axes[0, i].set_xlim(0, 1)
        
        # Purity distribution
        if len(purities[i]) > 0:
            axes[1, i].hist(purities[i], bins=50, color=palette[i], alpha=0.7)
            mean_pur = np.mean(purities[i])
            axes[1, i].axvline(mean_pur, color='red', linestyle='--',
                              label=f'mean pur={mean_pur:.2f}')
            axes[1, i].set_xlabel("Purity")
            axes[1, i].set_ylabel("Count")
            axes[1, i].legend(fontsize=8)
            axes[1, i].set_xlim(0, 1)
    
    axes[0, 0].set_title("Efficiency distribution per class", loc='left', fontsize=12, fontweight='bold')
    axes[1, 0].set_title("Purity distribution per class", loc='left', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "efficiency_purity_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Efficiency and purity plot saved: {save_path}")

def plot_roc_curves(targets, scores, class_names, save_dir):
    """Plot ROC curves for each class with AUC values"""
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Binarize targets for multi-class ROC
    from sklearn.preprocessing import label_binarize
    targets_bin = label_binarize(targets, classes=range(n_classes))
    
    auc_scores = {}
    
    for i in range(n_classes):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(targets_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = roc_auc
        
        # Plot
        axes[i].plot(fpr, tpr, color='blue', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{class_names[i]}')
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('ROC Curves per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "roc_curves_per_class.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved: {save_path}")
    
    return auc_scores

def report_dataset_splits(net):
    """Report dataset split information"""
    counts = {}
    for name in ["training_dataset", "validation_dataset", "testing_dataset"]:
        if hasattr(net, name):
            try:
                ds = getattr(net, name)
                n = len(ds)
                counts[name] = n
                print(f"{name}: {n} samples")
            except Exception as e:
                print(f"{name}: length unavailable ({e})")
    if counts:
        total = sum(counts.values())
        if total > 0:
            print("Split ratios:")
            for k, n in counts.items():
                print(f"  {k}: {n/total:.2%}")
    return counts

# ========== Load Model ========== 
print("="*60)
print("HPST MODEL EVALUATION - NOvA Dataset")
print("="*60)
print("\nLoading model...")
options = Options.load(OPTIONS_PATH)
options.testing_file = TESTING_FILE
options.num_dataloader_workers = 0

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
network = HeterogenousPointSetTrainer(options, num_objects=10)
network.load_state_dict(checkpoint["state_dict"])
network = network.eval()
for p in network.parameters():
    p.requires_grad_(False)
network = network.to(DEVICE)
print(f"✓ Model loaded on {DEVICE}")
print(f"✓ Model has {network.num_classes} semantic classes and {network.num_objects} object classes")

# ========== Dataset Information ==========
print("\n" + "="*60)
print("Dataset Split Summary")
print("="*60)
split_counts = report_dataset_splits(network)

# ========== Prepare DataLoader ==========
DATASET = network.testing_dataset
DATASET.return_index = True

dataloader_options = network.dataloader_options
dataloader_options["pin_memory"] = True if USE_CUDA else False
dataloader_options["num_workers"] = max(4, os.cpu_count() // 2)  
dataloader_options["batch_size"] = 512      
dataloader_options["drop_last"] = False

test_dataloader = network.dataloader(DATASET, **dataloader_options)
print(f"\n✓ Total batches in test set: {len(test_dataloader)}")

# ========== Debug: Check Class Information ========== (Line 375)
print("\n" + "="*60)
print("DEBUG: Dataset Class Information")
print("="*60)
print(f"Network num_classes: {network.num_classes}")

if hasattr(DATASET, 'class_names'):
    print(f"Dataset class_names: {DATASET.class_names}")
elif hasattr(DATASET, 'classes'):
    print(f"Dataset classes: {DATASET.classes}")
else:
    print("No class names found in dataset")

try:
    first_batch = next(iter(test_dataloader))
    targets1, targets2 = first_batch[4], first_batch[9]
    unique_labels_1 = targets1[targets1 != -1].unique() if targets1.numel() > 0 else torch.tensor([])
    unique_labels_2 = targets2[targets2 != -1].unique() if targets2.numel() > 0 else torch.tensor([])
    print(f"Target labels range view1: {unique_labels_1}")
    print(f"Target labels range view2: {unique_labels_2}")
    
    # Recreate dataloader
    test_dataloader = network.dataloader(DATASET, **dataloader_options)
except Exception as e:
    print(f"Warning: Could not inspect first batch - {e}")

print("="*60 + "\n")

# ========== Determine Class Names to Use ==========
current_names = get_current_class_names(DATASET, network.num_classes)
print(f"Current class names from dataset: {current_names}")

if network.num_classes == NUM_CLASSES:
    print(f"✓ Using NOvA class names ({NUM_CLASSES} classes)")
    class_names = NOVA_CLASS_NAMES
    palette = NOVA_PALETTE
    
    # Check if we need to remap labels
    need_remap = False
    for i, name in enumerate(current_names):
        if str(name).lower().strip() != NOVA_CLASS_NAMES[i].lower():
            need_remap = True
            break
    
    if need_remap:
        print("  Building label remapping...")
        label_remap = build_label_remap(current_names, NOVA_CLASS_NAMES)
        print(f"  Label mapping: {dict(enumerate(label_remap.tolist()))}")
    else:
        print("  No remapping needed, labels already match")
        label_remap = torch.arange(NUM_CLASSES, dtype=torch.long)
else:
    print(f"⚠ Warning: Dataset has {network.num_classes} classes, expected {NUM_CLASSES}")
    print("  Using generic class names")
    class_names = current_names
    palette = sns.color_palette("tab10", network.num_classes).as_hex()
    label_remap = torch.arange(network.num_classes, dtype=torch.long)

print(f"\nFinal class configuration:")
for i, (name, color) in enumerate(zip(class_names, palette)):
    print(f"  Class {i}: {name:15s} (color: {color})")
print()


# ========== Inference ========== (Line 412)
print("="*60)
print("Running Inference")
print("="*60)

all_predictions1, all_targets1, all_scores1 = [], [], []
all_predictions2, all_targets2, all_scores2 = [], [], []

examples_to_save = EXAMPLES_TO_SAVE
examples_saved = 0

with torch.inference_mode():
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
            break
        (
            ids,
            batches1, features1, coordinates1, targets1, object_targets1,
            batches2, features2, coordinates2, targets2, object_targets2
        ) = batch

        # Move to device (non_blocking + pin_memory更佳)
        batches1 = to_device(batches1); features1 = to_device(features1)
        coordinates1 = to_device(coordinates1); targets1 = to_device(targets1)
        batches2 = to_device(batches2); features2 = to_device(features2)
        coordinates2 = to_device(coordinates2); targets2 = to_device(targets2)

        # Mixed precision on GPU
        with torch.amp.autocast(enabled=USE_CUDA, dtype=torch.float16, device_type='cuda'):
            predictions1, object_predictions1, predictions2, object_predictions2 = network.forward(
                features1, coordinates1, batches1, features2, coordinates2, batches2
            )

        # Store results
        all_predictions1.append(torch.argmax(predictions1, dim=1).cpu())
        all_targets1.append(targets1.cpu())
        all_scores1.append(torch.softmax(predictions1, dim=1).cpu())

        all_predictions2.append(torch.argmax(predictions2, dim=1).cpu())
        all_targets2.append(targets2.cpu())
        all_scores2.append(torch.softmax(predictions2, dim=1).cpu())

        # Save example events
        try:
            if examples_saved < examples_to_save:
                b1 = _to_1d(batches1.cpu()); b2 = _to_1d(batches2.cpu())
                c1 = coordinates1.cpu(); c2 = coordinates2.cpu()
                y1 = _to_1d(targets1.cpu()); y2 = _to_1d(targets2.cpu())
                logit1 = predictions1.detach().cpu(); logit2 = predictions2.detach().cpu()
                
                cid1 = torch.argmax(object_predictions1.detach().cpu(), dim=1) if object_predictions1 is not None else None
                cid2 = torch.argmax(object_predictions2.detach().cpu(), dim=1) if object_predictions2 is not None else None

                bs = int(b1.max().item()) + 1 if b1.numel() > 0 else 0
                for local_idx in range(bs):
                    if examples_saved >= examples_to_save:
                        break
                    m1 = (b1 == local_idx) & (y1 != -1)
                    m2 = (b2 == local_idx) & (y2 != -1)
                    if not (m1.any() and m2.any()):
                        continue

                    # Apply majority voting within segments
                    pred_seg1 = segment_majority_labels(cid1[m1] if cid1 is not None else None, logit1[m1])
                    pred_seg2 = segment_majority_labels(cid2[m2] if cid2 is not None else None, logit2[m2])

                    # No remapping for visualization - use original labels
                    sample_id = safe_sample_id(ids, local_idx, batch_idx)
                    save_path = os.path.join("results", "hpst", f"hpst_example_event_{sample_id}.png")
                    plot_event(
                        sample_id,
                        c1[m1], y1[m1], pred_seg1,
                        c2[m2], y2[m2], pred_seg2,
                        class_names, palette, save_path
                    )
                    print(f"✓ Example event saved: {save_path}")
                    examples_saved += 1
        except Exception as e:
            print(f"[Warning] Visualization failed in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

# ========== Merge Results ==========
print("\nProcessing results...")
all_predictions = torch.cat(all_predictions1 + all_predictions2).numpy()
all_targets = torch.cat(all_targets1 + all_targets2).numpy()
all_scores = torch.cat(all_scores1 + all_scores2).numpy()

# Filter valid samples
valid_mask = (all_targets != -1) & (all_targets < len(class_names))
all_predictions = all_predictions[valid_mask]
all_targets = all_targets[valid_mask]
all_scores = all_scores[valid_mask]

print(f"✓ Total valid samples: {len(all_predictions)}")
print(f"✓ Unique target classes: {np.unique(all_targets)}")
print(f"✓ Unique predicted classes: {np.unique(all_predictions)}")

# ========== Metrics ==========
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

accuracy = accuracy_score(all_targets, all_predictions)
print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Per-class ROC-AUC
print("\n" + "-"*60)
print("Per-Class ROC-AUC Scores")
print("-"*60)
auc_scores = {} 
try:
    from sklearn.preprocessing import label_binarize
    targets_bin = label_binarize(all_targets, classes=range(len(class_names)))
    
    for i, cls_name in enumerate(class_names):
        try:
            if np.sum(targets_bin[:, i]) > 0:  # Only if class exists in test set
                cls_auc = roc_auc_score(targets_bin[:, i], all_scores[:, i])
                print(f"{cls_name:20s}: {cls_auc:.4f}")
                auc_scores[cls_name] = float(cls_auc)
            else:
                print(f"{cls_name:20s}: No samples in test set")
                auc_scores[cls_name] = 0.0
        except Exception as e:
            print(f"{cls_name:20s}: Cannot compute - {str(e)}")
            auc_scores[cls_name] = 0.0
    
    # Weighted average
    try:
        weighted_auc = roc_auc_score(all_targets, all_scores, multi_class='ovr', average='weighted')
        print(f"\n{'Weighted Average':20s}: {weighted_auc:.4f}")
    except:
        print(f"\n{'Weighted Average':20s}: Cannot compute")
except Exception as e:
    print(f"ROC-AUC computation failed: {str(e)}")

print("\n" + "-"*60)
print("Classification Report")
print("-"*60)
print(classification_report(all_targets, all_predictions, target_names=class_names, digits=4, zero_division=0))

# ========== Visualizations ==========
print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

output_dir = 'results/hpst'
os.makedirs(output_dir, exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title(f'Confusion Matrix\n(Overall Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved")
plt.close()

# 2. Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Percentage'}
)
plt.title('Normalized Confusion Matrix (Row-wise %)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=150, bbox_inches='tight')
print("✓ Normalized confusion matrix saved")
plt.close()

# 3. Per-Class Accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(14, 7))
bars = plt.bar(range(len(class_accuracies)), class_accuracies, 
               color=palette[:len(class_accuracies)], alpha=0.8, edgecolor='black')
plt.axhline(y=accuracy, color='red', linestyle='--', linewidth=2,
           label=f'Overall Accuracy: {accuracy:.2%}')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 1.1])
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, 
            f'{acc:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=150, bbox_inches='tight')
print("✓ Per-class accuracy saved")
plt.close()

# 4. ROC Curves per Class
if DO_ROC:
    print("Generating ROC curves...")
    auc_scores = plot_roc_curves(all_targets, all_scores, class_names, output_dir)

# 5. Efficiency and Purity Distribution (Figure 3 style)
print("Generating efficiency and purity distributions...")
plot_efficiency_purity(all_predictions, all_targets, class_names, palette, output_dir)

# ========== Summary Statistics ==========
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

unique, counts = np.unique(all_targets, return_counts=True)
print("\nClass Distribution in Test Set:")
for cls, count in zip(unique, counts):
    pct = count / len(all_targets) * 100
    cls_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
    print(f"  {cls_name:20s}: {count:7d} samples ({pct:5.2f}%)")

print("\n" + "-"*60)
print("Per-Class Performance Metrics")
print("-"*60)
print(f"{'Class':<20} {'Samples':<10} {'Accuracy':<12} {'AUC':<10}")
print("-"*60)
for i in range(len(class_names)):
    n_samples = cm.sum(axis=1)[i]
    n_correct = cm[i, i]
    acc = class_accuracies[i]
    auc_val = auc_scores.get(class_names[i], 0.0)
    print(f"{class_names[i]:<20} {n_samples:<10} {acc:>6.2%} ({n_correct}/{n_samples:>6}) {auc_val:>8.4f}")

# ========== Final Summary ========== (Line 672)
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\nNOvA Dataset - 6 Particle Classes:")
for i, name in enumerate(class_names):
    print(f"  Class {i}: {name}")
print(f"\nResults saved to '{output_dir}/' directory:")
print("  - confusion_matrix.png")
print("  - confusion_matrix_normalized.png")
print("  - class_accuracy.png")
if DO_ROC:
    print("  - roc_curves_per_class.png")
print("  - efficiency_purity_distribution.png")
print("  - hpst_example_event_*.png (example events)")
print("\n" + "="*60)