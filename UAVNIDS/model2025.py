"""
UAV Cyberattack Detection - Step 9: Model Training
====================================================
Models  : 1) Custom CNN (our own architecture)
          2) EfficientNetB0 (pretrained, transfer learning)
Framework: PyTorch
Classes  : Normal Traffic / Blackhole Attack / Wormhole Attack /
           Sybil Attack / Flooding Attack  (5-class)
Input    : preprocessed_output/images/  (64x64 RGB PNGs)
"""

import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "preprocessed_output", "images")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "model_output1")
META_PATH    = os.path.join(SCRIPT_DIR, "preprocessed_output", "meta.json")

IMG_SIZE     = 64
BATCH_SIZE   = 32
NUM_CLASSES  = 5                    # changed from 3
EPOCHS       = 30
LR           = 1e-4
PATIENCE     = 3
RANDOM_SEED  = 42
CLASS_NAMES  = [                    # changed from 3-class GPS names
    "Blackhole Attack",
    "Flooding Attack",
    "Normal Traffic",
    "Sybil Attack",
    "Wormhole Attack",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 62)
print("   UAVNIDS ATTACK DETECTION  |  Step 9: Model Training")
print("=" * 62)
print(f"\n  Device     : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ──────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────
print("\n[DATA] Setting up data loaders ...")

# ImageNet mean/std for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Train transform: augmentation to prevent 100% val accuracy
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Val/Test transform: no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = datasets.ImageFolder(
    os.path.join(IMAGES_DIR, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(IMAGES_DIR, "val"), transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(IMAGES_DIR, "test"), transform=eval_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True,
                          drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"  Train : {len(train_dataset):>6} images  |  {len(train_loader)} batches")
print(f"  Val   : {len(val_dataset):>6} images  |  {len(val_loader)} batches")
print(f"  Test  : {len(test_dataset):>6} images  |  {len(test_loader)} batches")
print(f"  Class mapping: {train_dataset.class_to_idx}")

# ──────────────────────────────────────────────────────────
# CLASS WEIGHTS  (from meta.json or compute on the fly)
# ──────────────────────────────────────────────────────────
train_counts = np.array([
    len(os.listdir(os.path.join(IMAGES_DIR, "train", c)))
    for c in sorted(os.listdir(os.path.join(IMAGES_DIR, "train")))
], dtype=np.float32)
class_weights = torch.tensor(
    train_counts.sum() / (NUM_CLASSES * train_counts),
    dtype=torch.float32
).to(DEVICE)
print(f"\n  Class weights : {class_weights.cpu().numpy()}")


# ══════════════════════════════════════════════════════════
# MODEL 1 — CUSTOM CNN
# ══════════════════════════════════════════════════════════
class UAVCustomCNN(nn.Module):

    def __init__(self, num_classes=5):             # changed default to 5
        super(UAVCustomCNN, self).__init__()

        def conv_block(in_ch, out_ch, dropout=0.3):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
            )

        self.features = nn.Sequential(
            conv_block(3,   32,  dropout=0.3),
            conv_block(32,  64,  dropout=0.3),
            conv_block(64,  128, dropout=0.4),
            conv_block(128, 256, dropout=0.4),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes),           # num_classes = 5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════
# MODEL 2 — EFFICIENTNETB0 (Transfer Learning)
# ══════════════════════════════════════════════════════════
def build_efficientnet(num_classes=5, freeze_until=0.7):    

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze bottom layers
    total_layers = len(list(model.parameters()))
    freeze_count = int(total_layers * freeze_until)
    for i, param in enumerate(model.parameters()):
        param.requires_grad = (i >= freeze_count)

    frozen   = sum(1 for p in model.parameters() if not p.requires_grad)
    unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNetB0: {frozen} layers frozen, {unfrozen} layers trainable")


    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),             
    )
    return model


# ══════════════════════════════════════════════════════════
# TRAINING ENGINE
# ══════════════════════════════════════════════════════════
def train_model(model, model_name, train_loader, val_loader,
                epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """
    Full training loop with:
    - CrossEntropyLoss with class weights
    - AdamW optimizer with weight decay
    - CosineAnnealing LR scheduler
    - Early stopping on val loss
    - Best model checkpoint saving
    """
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve     = 0
    start_time     = time.time()

    print(f"\n{'─'*62}")
    print(f"  Training: {model_name}")
    print(f"{'─'*62}")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} "
          f"{'Val Loss':<14} {'Val Acc':<12} {'LR'}")

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total * 100

        # ── VALIDATE ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs   = model(imgs)
                loss      = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds     = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total * 100

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  {epoch:<8} {train_loss:<14.4f} {train_acc:<14.2f} "
              f"{val_loss:<14.4f} {val_acc:<12.2f} {current_lr:.2e}")

        # ── EARLY STOPPING ──
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve     = 0
            torch.save(best_model_wts,
                       os.path.join(OUTPUT_DIR, f"{model_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\n  Training time : {total_time:.1f}s")
    print(f"  Best val loss : {best_val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history, total_time


# ══════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════
def evaluate_model(model, loader, model_name):
    """
    Full evaluation:
    - Accuracy, F1 per class, macro F1
    - Confusion matrix
    - ROC AUC (one-vs-rest)
    - Inference latency
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    latencies = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            t0 = time.time()
            outputs = model(imgs)
            latencies.append((time.time() - t0) / imgs.size(0) * 1000)

            probs  = torch.softmax(outputs, dim=1).cpu().numpy()
            preds  = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc     = accuracy_score(all_labels, all_preds) * 100
    f1_mac  = f1_score(all_labels, all_preds, average="macro") * 100
    f1_per  = f1_score(all_labels, all_preds, average=None) * 100
    auc     = roc_auc_score(all_labels, all_probs, multi_class="ovr") * 100
    latency = np.mean(latencies)
    cm      = confusion_matrix(all_labels, all_preds)

    # Use sorted class names to match ImageFolder ordering
    sorted_classes = sorted(os.listdir(os.path.join(IMAGES_DIR, "test")))

    print(f"\n{'─'*62}")
    print(f"  Results: {model_name}  (Test Set)")
    print(f"{'─'*62}")
    print(f"  Accuracy       : {acc:.2f}%")
    print(f"  Macro F1       : {f1_mac:.2f}%")
    print(f"  ROC AUC        : {auc:.2f}%")
    print(f"  Avg Latency    : {latency:.3f} ms/sample")
    print(f"\n  Per-class F1:")
    for cls, f1 in zip(sorted_classes, f1_per):
        print(f"    {cls:<25} : {f1:.2f}%")
    print(f"\n{classification_report(all_labels, all_preds, target_names=sorted_classes)}")

    return {
        "accuracy": acc, "macro_f1": f1_mac,
        "roc_auc": auc, "latency_ms": latency,
        "f1_per_class": dict(zip(sorted_classes, f1_per.tolist())),
        "confusion_matrix": cm.tolist(),
    }, cm, all_labels, all_probs


# ══════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════
def plot_learning_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss",
                 color="steelblue", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",
                 color="tomato", linewidth=2)
    axes[0].set_title(f"{model_name} — Loss Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train Acc",
                 color="steelblue", linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",
                 color="tomato", linewidth=2)
    axes[1].set_title(f"{model_name} — Accuracy Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots",
                f"{model_name}_learning_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(9, 8))             # larger for 5 classes
    sorted_classes = sorted(os.listdir(os.path.join(IMAGES_DIR, "test")))
    short_labels = [c.replace(" Attack", "").replace(" Traffic", "")
                    for c in sorted_classes]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_labels, yticklabels=short_labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots",
                f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()


def plot_comparison(results):
    metrics = ["accuracy", "macro_f1", "roc_auc"]
    labels  = ["Accuracy (%)", "Macro F1 (%)", "ROC AUC (%)"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#FF9800"]
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        vals = [results[mname][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=mname, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Comparison — Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "model_comparison.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN — RUN BOTH MODELS
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    all_results = {}

    # ── MODEL 1: Custom CNN ──
    print("\n" + "=" * 62)
    print("  MODEL 1: Custom CNN")
    print("=" * 62)
    custom_cnn = UAVCustomCNN(num_classes=NUM_CLASSES)
    total_params     = sum(p.numel() for p in custom_cnn.parameters())
    trainable_params = sum(p.numel() for p in custom_cnn.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    custom_cnn, history_cnn, time_cnn = train_model(
        custom_cnn, "CustomCNN", train_loader, val_loader
    )
    plot_learning_curves(history_cnn, "CustomCNN")
    results_cnn, cm_cnn, _, _ = evaluate_model(custom_cnn, test_loader, "CustomCNN")
    results_cnn["training_time_s"] = time_cnn
    plot_confusion_matrix(cm_cnn, "CustomCNN")
    all_results["CustomCNN"] = results_cnn

    # ── MODEL 2: EfficientNetB0 ──
    print("\n" + "=" * 62)
    print("  MODEL 2: EfficientNetB0 (Transfer Learning)")
    print("=" * 62)
    effnet = build_efficientnet(num_classes=NUM_CLASSES, freeze_until=0.7)
    total_params     = sum(p.numel() for p in effnet.parameters())
    trainable_params = sum(p.numel() for p in effnet.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    effnet, history_eff, time_eff = train_model(
        effnet, "EfficientNetB0", train_loader, val_loader
    )
    plot_learning_curves(history_eff, "EfficientNetB0")
    results_eff, cm_eff, _, _ = evaluate_model(effnet, test_loader, "EfficientNetB0")
    results_eff["training_time_s"] = time_eff
    plot_confusion_matrix(cm_eff, "EfficientNetB0")
    all_results["EfficientNetB0"] = results_eff

    # ── COMPARISON ──
    plot_comparison(all_results)

    # ── SAVE RESULTS ──
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 62)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<20} {'CustomCNN':>15} {'EfficientNetB0':>16}")
    print(f"  {'─'*52}")
    for metric in ["accuracy", "macro_f1", "roc_auc", "latency_ms", "training_time_s"]:
        unit = "ms" if "latency" in metric else ("s" if "time" in metric else "%")
        v1 = all_results["CustomCNN"][metric]
        v2 = all_results["EfficientNetB0"][metric]
        print(f"  {metric:<20} {v1:>14.2f}{unit}  {v2:>14.2f}{unit}")

    print(f"\n  Outputs saved in : {OUTPUT_DIR}")
    print(f"    ├── CustomCNN_best.pth")
    print(f"    ├── EfficientNetB0_best.pth")
    print(f"    ├── results.json")
    print(f"    └── plots/  (learning curves, confusion matrices, comparison)")
    print("=" * 62)
    print("\nDone!")