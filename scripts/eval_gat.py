import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from hpst.utils.options import Options
from hpst.trainers.gat_trainer import GATTrainer

# ========= One-line switch: set True for GPU, False for CPU =========
USE_CUDA = False  # <--- 修改这一处即可切换 GPU/CPU

# Optional: select which GPU when USE_CUDA=True
CUDA_DEVICE_INDEX = 0

# File paths
OPTIONS_PATH = "/home/houyh/HPST-Nova/config/gnn/gat_tune_nova.json"
CHECKPOINT_PATH = "/home/houyh/HPST-Nova/HPST/yz1y9c6g/checkpoints/last.ckpt"
TESTING_FILE = "/mnt/ironwolf_14t/users/ayankelevich/preprocessed_nova_miniprod6_1_cvnlabmaps.h5"

# ========= Device =========
if USE_CUDA and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_INDEX)
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
    USE_CUDA = False

def to_device(x):
    return x.to(DEVICE) if torch.is_tensor(x) else x

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

# ========== Load Model ==========
print("Loading model...")
options = Options.load(OPTIONS_PATH)
options.testing_file = TESTING_FILE
options.num_dataloader_workers = 0

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
network = GATTrainer(options)
network.load_state_dict(checkpoint["state_dict"])
network = network.eval()
for p in network.parameters():
    p.requires_grad_(False)
network = network.to(DEVICE)
print(f"Model loaded on {DEVICE}")

# ===== NEW: 打印 train/val/test 样本数与占比 =====
def _ds_path(ds):
    # 尝试获取底层文件路径（不同实现可能字段不同）
    for attr in ("file", "path", "filename", "h5_path"):
        if hasattr(ds, attr):
            return getattr(ds, attr)
    return "unknown"

def report_dataset_splits(net):
    counts = {}
    for name in ["training_dataset", "validation_dataset", "testing_dataset"]:
        if hasattr(net, name):
            try:
                ds = getattr(net, name)
                n = len(ds)
                counts[name] = n
                print(f"{name}: {n} samples from {_ds_path(ds)}")
            except Exception as e:
                print(f"{name}: length unavailable ({e})")
    if counts:
        total = sum(counts.values())
        if total > 0:
            print("Split ratios (over available splits):")
            for k, n in counts.items():
                print(f"  {k}: {n/total:.2%}")
    return counts

print("\n=== Dataset split summary ===")
split_counts = report_dataset_splits(network)
print("================================\n")

# ========== Prepare Data ==========
DATASET = network.validation_dataset
DATASET.return_index = True

dataloader_options = network.dataloader_options
dataloader_options["pin_memory"] = False
dataloader_options["num_workers"] = 0
dataloader_options["batch_size"] = 8
dataloader_options["drop_last"] = False

test_dataloader = network.dataloader(DATASET, **dataloader_options)
print(f"Total batches: {len(test_dataloader)}")

# ========== Inference accumulators ==========
print("\nRunning inference on test set...")
all_predictions1, all_targets1, all_scores1 = [], [], []
all_predictions2, all_targets2, all_scores2 = [], [], []

# ---------- Paper legend (9 classes, fixed order and colors) ----------
PAPER_CLASS_ORDER = [
    "Charged Kaon", "Neutral Pion", "Charged Pion", "Neutron",
    "Proton", "Muon", "Electron", "Other", "Background"
]
PAPER_PALETTE = [
    "#8E24AA", "#8D6E63", "#E53935", "#455A64",
    "#43A047", "#1E88E5", "#000000", "#FB8C00", "#FFFFFF"
]

def get_current_class_names(num_classes: int):
    for src in (getattr(DATASET, "class_names", None), getattr(network, "class_names", None)):
        if isinstance(src, (list, tuple)) and len(src) == num_classes:
            return list(src)
    return [f"Class {i}" for i in range(num_classes)]

def build_label_remap(current_names, target_names):
    cur2idx = {str(n).lower(): i for i, n in enumerate(current_names)}
    tgt2idx = {str(n).lower(): i for i, n in enumerate(target_names)}
    remap = torch.arange(len(current_names), dtype=torch.long)
    other_idx = tgt2idx.get("other", None)
    aliases = {
        "k+": "charged kaon", "kaon": "charged kaon", "kplus": "charged kaon",
        "pi0": "neutral pion", "pi0 (neutral pion)": "neutral pion",
        "pi+": "charged pion", "pion+": "charged pion", "piplus": "charged pion",
        "e-": "electron", "electron-": "electron", "electron shower": "electron",
        "mu-": "muon", "muon-": "muon",
        "p": "proton", "n": "neutron",
        "bg": "background", "bkg": "background"
    }
    for cur_i, cur_n in enumerate(current_names):
        key = str(cur_n).lower()
        j = tgt2idx.get(key, None)
        if j is None and key in aliases:
            j = tgt2idx.get(aliases[key], None)
        if j is None:
            j = other_idx if other_idx is not None else cur_i
        remap[cur_i] = j
    return remap

def apply_remap(labels_tensor: torch.Tensor, remap: torch.Tensor):
    if labels_tensor.dtype != torch.long:
        labels_tensor = labels_tensor.long()
    out = labels_tensor.clone()
    m = (out >= 0) & (out < remap.numel())
    out[m] = remap[out[m]]
    return out

def segment_majority_labels(cluster_ids_or_none: torch.Tensor, point_logits: torch.Tensor):
    point_pred = torch.argmax(point_logits, dim=1).cpu()
    if cluster_ids_or_none is None:
        return point_pred
    cid = cluster_ids_or_none.cpu().view(-1)
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
    c = coords.cpu().numpy()
    if c.shape[1] >= 3:
        return c[:, 2], c[:, 0]  # Z, X
    return c[:, 0], c[:, 1]

def extract_yz(coords: torch.Tensor):
    c = coords.cpu().numpy()
    if c.shape[1] >= 3:
        return c[:, 2], c[:, 1]  # Z, Y
    return c[:, 0], c[:, 1]

def plot_event(sample_id, coords1, y_true1, y_pred1, coords2, y_true2, y_pred2,
               class_names, palette, save_path):
    colors = np.array(palette)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    zx, zy = extract_xz(coords1)
    axes[0, 0].scatter(zx, zy, c=colors[y_true1.numpy()], s=2, linewidths=0)
    axes[0, 0].set_title(f"XZ, true label, sample {sample_id}")
    axes[0, 0].set_xlabel("Z"); axes[0, 0].set_ylabel("X")

    zy_z, zy_y = extract_yz(coords2)
    axes[0, 1].scatter(zy_z, zy_y, c=colors[y_true2.numpy()], s=2, linewidths=0)
    axes[0, 1].set_title(f"YZ, true label, sample {sample_id}")
    axes[0, 1].set_xlabel("Z"); axes[0, 1].set_ylabel("Y")

    zx, zy = extract_xz(coords1)
    axes[1, 0].scatter(zx, zy, c=colors[y_pred1.numpy()], s=2, linewidths=0)
    axes[1, 0].set_title(f"XZ, prediction, sample {sample_id}")
    axes[1, 0].set_xlabel("Z"); axes[1, 0].set_ylabel("X")

    zy_z, zy_y = extract_yz(coords2)
    axes[1, 1].scatter(zy_z, zy_y, c=colors[y_pred2.numpy()], s=2, linewidths=0)
    axes[1, 1].set_title(f"YZ, prediction, sample {sample_id}")
    axes[1, 1].set_xlabel("Z"); axes[1, 1].set_ylabel("Y")

    handles = [Line2D([0], [0], marker='o', color='w', label=class_names[i],
                      markerfacecolor=palette[i], markersize=8)
               for i in range(len(class_names))]
    axes[0, 1].legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# 保存示例事件数量
examples_to_save = 2
examples_saved = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        (
            ids,
            batches1, features1, coordinates1, targets1, object_targets1,
            batches2, features2, coordinates2, targets2, object_targets2
        ) = batch

        # Move tensors to device
        batches1 = to_device(batches1); features1 = to_device(features1)
        coordinates1 = to_device(coordinates1); targets1 = to_device(targets1)
        batches2 = to_device(batches2); features2 = to_device(features2)
        coordinates2 = to_device(coordinates2); targets2 = to_device(targets2)

        # Forward pass
        predictions1, object_predictions1, predictions2, object_predictions2 = network.forward(
            features1, coordinates1, batches1, features2, coordinates2, batches2
        )

        # Store results (on CPU)
        all_predictions1.append(torch.argmax(predictions1, dim=1).cpu())
        all_targets1.append(targets1.cpu())
        all_scores1.append(torch.softmax(predictions1, dim=1).cpu())

        all_predictions2.append(torch.argmax(predictions2, dim=1).cpu())
        all_targets2.append(targets2.cpu())
        all_scores2.append(torch.softmax(predictions2, dim=1).cpu())

        # ====== Figure 4 风格：两例事件可视化（XZ/YZ） ======
        try:
            if examples_saved < examples_to_save:
                b1 = _to_1d(batches1.cpu()); b2 = _to_1d(batches2.cpu())
                c1 = coordinates1.cpu(); c2 = coordinates2.cpu()
                y1 = _to_1d(targets1.cpu()); y2 = _to_1d(targets2.cpu())
                logit1 = predictions1.detach().cpu(); logit2 = predictions2.detach().cpu()
                # 先从 object_predictions logits 取 argmax，得到每点的 cluster id
                cid1 = torch.argmax(object_predictions1.detach().cpu(), dim=1) if object_predictions1 is not None else None
                cid2 = torch.argmax(object_predictions2.detach().cpu(), dim=1) if object_predictions2 is not None else None
                
                num_classes = logit1.shape[1]
                current_names = get_current_class_names(num_classes)
                if num_classes == len(PAPER_CLASS_ORDER):
                    remap = build_label_remap(current_names, PAPER_CLASS_ORDER)
                    palette = PAPER_PALETTE
                    class_names = PAPER_CLASS_ORDER
                else:
                    remap = torch.arange(num_classes, dtype=torch.long)
                    class_names = current_names
                    palette = sns.color_palette("tab20", num_classes).as_hex()

                bs = int(b1.max().item()) + 1 if b1.numel() > 0 else 0
                for local_idx in range(bs):
                    if examples_saved >= examples_to_save:
                        break
                    m1 = (b1 == local_idx) & (y1 != -1)
                    m2 = (b2 == local_idx) & (y2 != -1)
                    if not (m1.any() and m2.any()):
                        continue

                    pred_seg1 = segment_majority_labels(cid1[m1] if cid1 is not None else None, logit1[m1])
                    pred_seg2 = segment_majority_labels(cid2[m2] if cid2 is not None else None, logit2[m2])

                    y1_map = apply_remap(y1[m1], remap)
                    y2_map = apply_remap(y2[m2], remap)
                    pred1_map = apply_remap(pred_seg1, remap)
                    pred2_map = apply_remap(pred_seg2, remap)

                    sample_id = safe_sample_id(ids, local_idx, batch_idx)
                    save_path = os.path.join("results", "gat", f"gat_example_event_{sample_id}.png")
                    plot_event(
                        sample_id,
                        c1[m1], y1_map, pred1_map,
                        c2[m2], y2_map, pred2_map,
                        class_names, palette, save_path
                    )
                    print(f"✓ Example saved: {save_path}")
                    examples_saved += 1
        except Exception as e:
            print(f"[Warn] Visualization failed in batch {batch_idx}: {e}")

# ========== Merge Results ==========
print("\nProcessing results...")
all_predictions = torch.cat(all_predictions1 + all_predictions2).numpy()
all_targets = torch.cat(all_targets1 + all_targets2).numpy()
all_scores = torch.cat(all_scores1 + all_scores2).numpy()

valid_mask = (all_targets != -1)
all_predictions = all_predictions[valid_mask]
all_targets = all_targets[valid_mask]
all_scores = all_scores[valid_mask]
print(f"Total valid samples: {len(all_predictions)}")

# ========== Metrics ==========
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

accuracy = accuracy_score(all_targets, all_predictions)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

try:
    roc_auc = roc_auc_score(all_targets, all_scores, multi_class='ovr', average='weighted')
    print(f"Weighted ROC-AUC: {roc_auc:.4f}")
except Exception as e:
    print(f"ROC-AUC: Cannot compute - {str(e)}")

print("\nDetailed Classification Report:")
print(classification_report(
    all_targets,
    all_predictions,
    target_names=[f"Class {i}" for i in range(len(np.unique(all_targets)))]
))

# ========== Visualization ==========
print("\nGenerating visualizations...")
os.makedirs('results/gat', exist_ok=True)

cm = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[f"Class {i}" for i in range(cm.shape[0])],
    yticklabels=[f"Class {i}" for i in range(cm.shape[0])]
)
plt.title(f'Confusion Matrix\n(Overall Accuracy: {accuracy:.2%})', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/gat/gat_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved: results/gat/gat_confusion_matrix.png")
plt.close()

class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(class_accuracies)), class_accuracies, color='steelblue', alpha=0.8)
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.2%}')
plt.xlabel('Class'); plt.ylabel('Accuracy'); plt.title('Per-Class Accuracy')
plt.ylim([0, 1.1])
plt.xticks(range(len(class_accuracies)), [f"Class {i}" for i in range(len(class_accuracies))])
plt.legend(); plt.grid(True, alpha=0.3, axis='y')
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h, f'{acc:.2%}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('results/gat/gat_class_accuracy.png', dpi=150, bbox_inches='tight')
print("✓ Per-class accuracy saved: results/gat/gat_class_accuracy.png")
plt.close()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
    xticklabels=[f"Class {i}" for i in range(cm.shape[0])],
    yticklabels=[f"Class {i}" for i in range(cm.shape[0])]
)
plt.title('Normalized Confusion Matrix (Row-wise %)', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/gat/gat_confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
print("✓ Normalized confusion matrix saved: results/gat/gat_confusion_matrix_normalized.png")
plt.close()

# ========== Summary ==========
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
unique, counts = np.unique(all_targets, return_counts=True)
print("\nClass Distribution in Test Set:")
for cls, count in zip(unique, counts):
    pct = count / len(all_targets) * 100
    print(f"  Class {cls}: {count:6d} samples ({pct:5.2f}%)")

print("\nPer-Class Metrics:")
for i, acc in enumerate(class_accuracies):
    n_samples = cm.sum(axis=1)[i]
    n_correct = cm[i, i]
    print(f"  Class {i}: Accuracy = {acc:.4f} ({n_correct}/{n_samples} correct)")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\nResults saved to 'results/gat/' directory:")
print("  - gat_example_event_*.png (Figure 4-style samples)")
print("  - gat_confusion_matrix.png")
print("  - gat_confusion_matrix_normalized.png")
print("  - gat_class_accuracy.png")