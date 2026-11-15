
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH          = os.path.join(SCRIPT_DIR, "uav_cleaned1.csv")
OUTPUT_DIR         = os.path.join(SCRIPT_DIR, "preprocessed_output")
PLOTS_DIR          = os.path.join(OUTPUT_DIR, "plots")
IMAGES_DIR         = os.path.join(OUTPUT_DIR, "images")
DATA_OUT_DIR       = os.path.join(OUTPUT_DIR, "data")

RANDOM_SEED        = 42
N_PCA_COMPONENTS   = 12            # 18 features → 12 PCs (intentionally < max)
TEST_SIZE          = 0.15
VAL_SIZE           = 0.1765        # of remaining 85% → ~15% overall
TARGET_COL         = "label"

CLASS_NAMES = [
    "Normal Traffic",
    "Blackhole Attack",
    "Wormhole Attack",
    "Sybil Attack",
    "Flooding Attack",
]
COLORS = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#4CAF50"]

# Grid layout for image conversion
# 12 PCs split across 3 channels → 4 PCs per channel → 2×2 grid per channel
GRID_H, GRID_W  = 2, 2
N_SLOTS         = max(GRID_H * GRID_W * 3, N_PCA_COMPONENTS)   # = 12
DISPLAY_SIZE    = (64, 64)

for d in [OUTPUT_DIR, PLOTS_DIR, IMAGES_DIR, DATA_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(RANDOM_SEED)

print("=" * 62)
print("   UAVNIDS ATTACK DETECTION PIPELINE  |  Steps 2 → 8")
print("   (SPLIT-FIRST — no data leakage)")
print(f"   Dataset    : UAVNIDS-2025  |  5 classes  |  18 features")
print(f"   PCA        : {N_PCA_COMPONENTS} components (fixed)")
print("=" * 62)


# ──────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────
print(f"\n[LOAD] Reading uav_cleaned.csv ...")
df = pd.read_csv(DATA_PATH)
print(f"       Shape : {df.shape}")

feature_cols = [c for c in df.columns if c != TARGET_COL]
X_raw = df[feature_cols].copy()
y_raw = df[TARGET_COL].copy()

print(f"       Features : {len(feature_cols)}")
print(f"       Classes  : {sorted(y_raw.unique().tolist())}")

le = LabelEncoder()
le.fit(CLASS_NAMES)
y_enc = le.transform(y_raw)


# ══════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════
print("\n[STEP 2] Exploratory Data Analysis ...")

# --- 2a. Class distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
counts = y_raw.value_counts().reindex(CLASS_NAMES)
bars = ax.bar(CLASS_NAMES, counts.values, color=COLORS, edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 60,
            f"{val}\n({val/len(y_raw)*100:.1f}%)",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Class Distribution (Full Dataset)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sample Count")
ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
ax.set_ylim(0, counts.max() * 1.22)
ax.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
plt.close()

# --- 2b. Feature distributions ---
key_features = [
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s", "PacketDropRate", "AverageHopCount"
]
key_features = [f for f in key_features if f in df.columns]

if key_features:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        for cls, col in zip(CLASS_NAMES, COLORS):
            vals = df[df[TARGET_COL] == cls][feat].dropna()
            axes[i].hist(vals, bins=40, alpha=0.55, color=col,
                         label=cls.replace(" Attack", "").replace(" Traffic", ""),
                         density=True)
        axes[i].set_title(feat, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=8)
        axes[i].set_ylabel("Density", fontsize=8)
        axes[i].legend(fontsize=6)
        axes[i].grid(alpha=0.2)
    for j in range(len(key_features), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Key Feature Distributions by Attack Type (Raw)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), dpi=150)
    plt.close()

# --- 2c. Correlation heatmap ---
fig, ax = plt.subplots(figsize=(14, 12))
corr = X_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, linewidths=0.5,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title(f"Feature Correlation Heatmap ({len(feature_cols)} features)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

print("       Plots saved: class_distribution, feature_distributions, correlation_heatmap")

# --- 2d. Box plots ---
if key_features:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    short_names = [c.replace(" Attack", "").replace(" Traffic", "") for c in CLASS_NAMES]
    for i, feat in enumerate(key_features):
        data_by_class = [df[df[TARGET_COL] == cls][feat].dropna().values
                         for cls in CLASS_NAMES]
        bp = axes[i].boxplot(data_by_class, patch_artist=True,
                             medianprops=dict(color="black", linewidth=2))
        for patch, col in zip(bp["boxes"], COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        axes[i].set_xticks(range(1, len(CLASS_NAMES)+1))
        axes[i].set_xticklabels(short_names, fontsize=7, rotation=15)
        axes[i].set_title(feat, fontsize=9, fontweight="bold")
        axes[i].set_ylabel("Value", fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)
    for j in range(len(key_features), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Key Feature Box Plots by Attack Type (Raw)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_boxplots.png"), dpi=150)
    plt.close()
    print("       Plot saved: feature_boxplots")


# ══════════════════════════════════════════════════════════
# STEP 5 — STRATIFIED SPLIT  (must happen FIRST)
# ══════════════════════════════════════════════════════════
print("\n[STEP 5] Stratified Split (FIRST — before any fitting) ...")

X_trainval, X_test_raw, y_trainval, y_test = train_test_split(
    X_raw, y_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_enc
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

print(f"       Train : {len(X_train_raw):>6} rows  ({len(X_train_raw)/len(X_raw)*100:.1f}%)")
print(f"       Val   : {len(X_val_raw):>6} rows  ({len(X_val_raw)/len(X_raw)*100:.1f}%)")
print(f"       Test  : {len(X_test_raw):>6} rows  ({len(X_test_raw)/len(X_raw)*100:.1f}%)")
print(f"       Label mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")
print("\n  *** Scaler/PCA will now be fit on TRAIN only, then applied to val/test ***")


# ══════════════════════════════════════════════════════════
# STEP 3 — ROBUST SCALER  (fit on train only)
# ══════════════════════════════════════════════════════════
print("\n[STEP 3] RobustScaler — fit on train, transform all splits ...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)
X_test_scaled  = scaler.transform(X_test_raw)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_val_scaled   = pd.DataFrame(X_val_scaled,   columns=feature_cols)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_cols)

print(f"       Fit on train ({len(X_train_raw)} rows). Val/test transformed with same params.")
print(f"       Train median ~ 0 check: {X_train_scaled.median().abs().mean():.4f}")

for name, Xs, ys in [("train_scaled", X_train_scaled, y_train),
                      ("val_scaled",   X_val_scaled,   y_val),
                      ("test_scaled",  X_test_scaled,  y_test)]:
    out = Xs.copy()
    out["label"] = ys
    out["attack_type"] = le.inverse_transform(ys)
    out.to_csv(os.path.join(DATA_OUT_DIR, f"{name}.csv"), index=False)
    print(f"       Saved {name}.csv -> {out.shape}")


# ══════════════════════════════════════════════════════════
# STEP 4 — PCA  (fit on train only, fixed 12 components)
# ══════════════════════════════════════════════════════════
print(f"\n[STEP 4] PCA (fixed {N_PCA_COMPONENTS} components) — fit on train only ...")

n_components = N_PCA_COMPONENTS

pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

cumvar_kept = float(np.sum(pca.explained_variance_ratio_))
print(f"       Components selected : {n_components}  (fixed)")
print(f"       Variance explained  : {cumvar_kept*100:.2f}%")

pc_cols = [f"PC{i+1}" for i in range(n_components)]
X_train_pca = pd.DataFrame(X_train_pca, columns=pc_cols)
X_val_pca   = pd.DataFrame(X_val_pca,   columns=pc_cols)
X_test_pca  = pd.DataFrame(X_test_pca,  columns=pc_cols)

# Scree plot
cumvar_all = np.cumsum(pca.explained_variance_ratio_)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, n_components+1), cumvar_all*100, marker="o",
             color="steelblue", linewidth=2)
axes[0].axvline(n_components, color="green", linestyle="--",
                label=f"n = {n_components} (fixed)")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Variance (%)")
axes[0].set_title("PCA — Cumulative Explained Variance (train)")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].bar(range(1, n_components+1),
            pca.explained_variance_ratio_*100,
            color="steelblue", alpha=0.8)
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Explained Variance (%)")
axes[1].set_title(f"PCA — Individual Variance ({n_components} PCs, train)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_scree.png"), dpi=150)
plt.close()
print(f"       Scree plot saved.")

for name, Xp, ys in [("train_pca", X_train_pca, y_train),
                      ("val_pca",   X_val_pca,   y_val),
                      ("test_pca",  X_test_pca,  y_test)]:
    out = Xp.copy()
    out["label"] = ys
    out["attack_type"] = le.inverse_transform(ys)
    out.to_csv(os.path.join(DATA_OUT_DIR, f"{name}.csv"), index=False)
    print(f"       Saved {name}.csv -> {out.shape}")

# --- 4a. t-SNE ---
print("       Running t-SNE on train PCA (this may take ~30s) ...")
tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=40,
            max_iter=1000, learning_rate="auto", init="pca")
max_tsne = 3000
if len(X_train_pca) > max_tsne:
    idx = np.random.choice(len(X_train_pca), max_tsne, replace=False)
    X_tsne_in = X_train_pca.values[idx]
    y_tsne_in = y_train[idx]
else:
    X_tsne_in = X_train_pca.values
    y_tsne_in = y_train

X_tsne = tsne.fit_transform(X_tsne_in)

fig, ax = plt.subplots(figsize=(9, 7))
for cls_idx, (cls_name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
    mask = y_tsne_in == cls_idx
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=col, label=cls_name, alpha=0.6, s=15, linewidths=0)
ax.set_title("t-SNE of PCA Components (Train Set)", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=9, markerscale=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "tsne_pca.png"), dpi=150)
plt.close()
print("       Plot saved: tsne_pca")

# --- 4b. Pairplot top 5 PCs ---
print("       Generating pairplot of top 5 PCs ...")
top_k = min(5, n_components)
pair_df = X_train_pca.iloc[:, :top_k].copy()
pair_df["Class"] = le.inverse_transform(y_train)

max_pair = 1500
samples_per_class = max_pair // len(CLASS_NAMES)
subsets = []
for cls in CLASS_NAMES:
    subset = pair_df[pair_df["Class"] == cls]
    if len(subset) > samples_per_class:
        subset = subset.sample(samples_per_class, random_state=RANDOM_SEED)
    subsets.append(subset)
pair_df = pd.concat(subsets, ignore_index=True)

palette = {cls: col for cls, col in zip(CLASS_NAMES, COLORS)}
pg = sns.pairplot(pair_df, hue="Class", palette=palette,
                  plot_kws=dict(alpha=0.4, s=12, linewidths=0),
                  diag_kind="kde", corner=True)
pg.figure.suptitle("Pairplot — Top 5 PCA Components (Train)", y=1.01,
                   fontsize=13, fontweight="bold")
pg.figure.savefig(os.path.join(PLOTS_DIR, "pairplot_pca.png"), dpi=150,
                  bbox_inches="tight")
plt.close()
print("       Plot saved: pairplot_pca")


# ══════════════════════════════════════════════════════════
# STEP 6 — SMOTE + CLASS WEIGHTS  (train only)
# ══════════════════════════════════════════════════════════
print("\n[STEP 6] SMOTE oversampling (train only) ...")

print("       Before SMOTE:")
for cls_idx, cls_name in enumerate(le.classes_):
    cnt = np.sum(y_train == cls_idx)
    print(f"         {cls_name:<20} {cnt}")

from collections import Counter
_train_counts  = Counter(y_train.tolist())
_max_count     = int(max(_train_counts.values()))
_sampling_strategy = {
    int(k): _max_count
    for k, v in _train_counts.items()
    if int(v) < _max_count
}
print(f"       Majority class count : {_max_count}")
print(f"       Sampling strategy    : {_sampling_strategy}")

smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5,
              sampling_strategy=_sampling_strategy)
X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train)
print("       Oversampler used : SMOTE")

print("       After SMOTE:")
for cls_idx, cls_name in enumerate(le.classes_):
    cnt = np.sum(y_train_res == cls_idx)
    print(f"         {cls_name:<20} {cnt}")

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_res),
    y=y_train_res
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
print(f"       Class weights : {class_weights}")

# Before/after plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
before = [np.sum(y_train == i) for i in range(len(CLASS_NAMES))]
after  = [np.sum(y_train_res == i) for i in range(len(CLASS_NAMES))]
short  = [c.replace(" Attack", "").replace(" Traffic", "") for c in CLASS_NAMES]

for ax, vals, title in zip(axes,
                            [before, after],
                            ["Before SMOTE (Train)", "After SMOTE (Train)"]):
    bars = ax.bar(short, vals, color=COLORS, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30, str(v),
                ha="center", fontweight="bold", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_balance.png"), dpi=150)
plt.close()
print(f"       Class balance plot saved.")

for name, Xp, ys in [("train", X_train_res, y_train_res),
                      ("val",   X_val_pca,   y_val),
                      ("test",  X_test_pca,  y_test)]:
    cols = [f"PC{i+1}" for i in range(n_components)]
    arr  = Xp.values if hasattr(Xp, 'values') else Xp
    out  = pd.DataFrame(arr, columns=cols)
    out["label"] = ys
    out["attack_type"] = le.inverse_transform(ys)
    out.to_csv(os.path.join(DATA_OUT_DIR, f"{name}.csv"), index=False)
    print(f"       Saved {name}.csv -> {out.shape}")


# ══════════════════════════════════════════════════════════
# STEP 7 — IMAGE CONVERSION
# ══════════════════════════════════════════════════════════
print("\n[STEP 7] Converting PC values → RGB images ...")


def pcs_to_image(pc_row: np.ndarray, n_comp: int) -> np.ndarray:
    """Map 12 PCA values → 3-channel 2×2 grid → upscale to 64×64 RGB."""
    padded = np.zeros(N_SLOTS)
    padded[:n_comp] = pc_row
    channels = []
    for ch in range(3):
        chunk    = padded[ch * GRID_H * GRID_W : (ch+1) * GRID_H * GRID_W]
        chunk_2d = chunk.reshape(GRID_H, GRID_W)
        mn, mx   = chunk_2d.min(), chunk_2d.max()
        if mx - mn > 1e-8:
            chunk_norm = (chunk_2d - mn) / (mx - mn) * 255
        else:
            chunk_norm = np.zeros_like(chunk_2d)
        channels.append(chunk_norm.astype(np.uint8))
    return np.stack(channels, axis=-1)   # (2, 2, 3)


def save_images(X_arr, y_arr, split_name):
    counts = {c: 0 for c in CLASS_NAMES}
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(IMAGES_DIR, split_name, cls), exist_ok=True)
    X_np = X_arr.values if hasattr(X_arr, 'values') else X_arr
    for row, label in zip(X_np, y_arr):
        cls_name = le.inverse_transform([label])[0]
        img_arr  = pcs_to_image(row, n_components)
        img      = Image.fromarray(img_arr, mode="RGB")
        img      = img.resize(DISPLAY_SIZE, Image.NEAREST)
        fname    = f"{cls_name.replace(' ', '_')}_{counts[cls_name]:05d}.png"
        img.save(os.path.join(IMAGES_DIR, split_name, cls_name, fname))
        counts[cls_name] += 1
    total = sum(counts.values())
    print(f"       {split_name:<6} : {total} images")
    for k, v in counts.items():
        print(f"                {k:<22} {v}")


save_images(X_train_res, y_train_res, "train")
save_images(X_val_pca,   y_val,       "val")
save_images(X_test_pca,  y_test,      "test")

# Sample image grid (5 rows = 5 classes, 5 cols = 5 samples each)
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
for row_i, cls in enumerate(CLASS_NAMES):
    img_folder = os.path.join(IMAGES_DIR, "train", cls)
    img_files  = sorted(os.listdir(img_folder))[:5]
    for col_i, fname in enumerate(img_files):
        img = Image.open(os.path.join(img_folder, fname))
        axes[row_i][col_i].imshow(img)
        axes[row_i][col_i].axis("off")
        if col_i == 0:
            short_label = cls.replace(" Attack", "").replace(" Traffic", "")
            axes[row_i][col_i].set_ylabel(
                short_label, fontsize=9, fontweight="bold",
                rotation=90, labelpad=40
            )
plt.suptitle("Sample Images per Class (64×64 RGB)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=150)
plt.close()
print(f"       Sample images plot saved.")

# --- 7a. Mean image per class ---
print("       Generating mean image per class ...")
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for col_i, cls in enumerate(CLASS_NAMES):
    img_folder = os.path.join(IMAGES_DIR, "train", cls)
    arrays = [
        np.array(Image.open(os.path.join(img_folder, f))).astype(np.float32)
        for f in sorted(os.listdir(img_folder))
    ]
    mean_img = np.mean(arrays, axis=0).astype(np.uint8)
    axes[col_i].imshow(mean_img)
    short = cls.replace(" Attack", "").replace(" Traffic", "")
    axes[col_i].set_title(short, fontsize=10, fontweight="bold")
    axes[col_i].axis("off")
plt.suptitle("Mean Image per Class (Train, 64×64 RGB)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mean_image_per_class.png"), dpi=150)
plt.close()
print("       Plot saved: mean_image_per_class")

# --- 7b. Variance map per class ---
print("       Generating image variance map per class ...")
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for col_i, cls in enumerate(CLASS_NAMES):
    img_folder = os.path.join(IMAGES_DIR, "train", cls)
    arrays = [
        np.array(Image.open(os.path.join(img_folder, f))).astype(np.float32)
        for f in sorted(os.listdir(img_folder))
    ]
    var_img  = np.var(arrays, axis=0)
    var_gray = var_img.mean(axis=-1)
    im = axes[col_i].imshow(var_gray, cmap="hot")
    short = cls.replace(" Attack", "").replace(" Traffic", "")
    axes[col_i].set_title(short, fontsize=10, fontweight="bold")
    axes[col_i].axis("off")
    plt.colorbar(im, ax=axes[col_i], fraction=0.046, pad=0.04)
plt.suptitle("Pixel Variance Map per Class (Train)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "variance_map_per_class.png"), dpi=150)
plt.close()
print("       Plot saved: variance_map_per_class")


# ══════════════════════════════════════════════════════════
# STEP 8 — AUGMENTATION STRATEGY
# ══════════════════════════════════════════════════════════
print("\n[STEP 8] Augmentation strategy defined ...")
augmentation_config = {
    "horizontal_flip"    : True,
    "vertical_flip"      : False,
    "rotation_range"     : 0,
    "zoom_range"         : 0,
    "width_shift_range"  : 0,
    "height_shift_range" : 0,
}
print("       Type   : Horizontal flip only (applied at training time)")
print("       Reason : PCA images are data-encoded, not natural photos.")
print("                Heavy augmentation distorts PC relationships.")


# ══════════════════════════════════════════════════════════
# SAVE METADATA
# ══════════════════════════════════════════════════════════
print("\n[SAVE] Writing meta.json ...")
meta = {
    "dataset"                : "UAVNIDS-2025",
    "pipeline_order"         : ["split", "scale", "pca", "smote", "image_convert"],
    "leakage_free"           : True,
    "label_mapping"          : {cls: int(le.transform([cls])[0]) for cls in le.classes_},
    "n_features_input"       : len(feature_cols),
    "n_pca_components"       : int(n_components),
    "pca_variance_kept"      : cumvar_kept,
    "pca_mode"               : "fixed",
    "image_size_display"     : list(DISPLAY_SIZE),
    "grid_shape"             : [GRID_H, GRID_W],
    "channels"               : 3,
    "class_weights"          : class_weights,
    "augmentation"           : augmentation_config,
    "splits" : {
        "train" : int(len(X_train_res)),
        "val"   : int(len(X_val_pca)),
        "test"  : int(len(X_test_pca)),
    },
    "class_counts_train_after_smote" : {
        le.inverse_transform([u])[0]: int(c)
        for u, c in zip(*np.unique(y_train_res, return_counts=True))
    }
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("   PIPELINE COMPLETE — SUMMARY")
print("=" * 62)
print(f"  Dataset            : UAVNIDS-2025")
print(f"  Pipeline order     : Split → Scale → PCA → SMOTE → Images")
print(f"  Data leakage       : NONE (scaler/PCA fit on train only)")
print(f"  Input features     : {len(feature_cols)}")
print(f"  After PCA          : {n_components}  (fixed, {cumvar_kept*100:.2f}% variance)")
print(f"  Image size         : {DISPLAY_SIZE[0]}×{DISPLAY_SIZE[1]} RGB PNG")
print(f"  Train images (SMOTE)  : {len(X_train_res)}")
print(f"  Val images            : {len(X_val_pca)}")
print(f"  Test images           : {len(X_test_pca)}")
print(f"  Class weights         : {class_weights}")
print(f"  Output directory   : {OUTPUT_DIR}")
print(f"    ├── data/   *_scaled.csv, *_pca.csv, train.csv, val.csv, test.csv")
print(f"    ├── images/ train/ val/ test/")
print(f"    ├── plots/  11 plots")
print(f"    └── meta.json")
print("=" * 62)
print("\nReady for Step 9 — CNN + Transfer Learning!")
print("  Remember to update Model.py:")
print("    NUM_CLASSES = 5")
print("    CLASS_NAMES = ['Normal Traffic', 'Blackhole Attack',")
print("                   'Wormhole Attack', 'Sybil Attack', 'Flooding Attack']")