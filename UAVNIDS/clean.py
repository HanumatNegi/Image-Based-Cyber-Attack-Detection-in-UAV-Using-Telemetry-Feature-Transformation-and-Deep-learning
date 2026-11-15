

import os
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UAVNIDS25.csv")    # change path if needed
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uav_cleaned1.csv")

COLS_TO_DROP = ["FlowID", "SrcAddr", "DstAddr", "Protocol"]
TARGET_COL   = "label"

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  UAV Dataset — Column Removal & Cleaning")
print("=" * 60)

print(f"\n[LOAD] Reading {INPUT_PATH} ...")
df = pd.read_csv(INPUT_PATH)
print(f"       Raw shape : {df.shape}")

# ── SANITY CHECKS ─────────────────────────────────────────────────────────────
print("\n[CHECK] Pre-removal sanity checks ...")

# Confirm zero-variance / non-numeric columns we plan to drop
for col in COLS_TO_DROP:
    if col not in df.columns:
        print(f"  WARNING: '{col}' not found in dataset — skipping.")
        continue
    n_unique = df[col].nunique()
    dtype    = df[col].dtype
    print(f"  {col:<12}  dtype={str(dtype):<10}  unique_values={n_unique}")

# Null check before drop
nulls_before = df.isnull().sum().sum()
print(f"\n  Total nulls in raw data : {nulls_before}")

# ── DROP COLUMNS ─────────────────────────────────────────────────────────────
existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
df.drop(columns=existing_drops, inplace=True)
print(f"\n[DROP] Removed columns  : {existing_drops}")
print(f"       Shape after drop : {df.shape}")

# ── NULL HANDLING ─────────────────────────────────────────────────────────────
nulls_after = df.isnull().sum().sum()
if nulls_after > 0:
    print(f"\n[NULLS] Found {nulls_after} nulls — dropping affected rows ...")
    df.dropna(inplace=True)
    print(f"        Shape after dropna : {df.shape}")
else:
    print(f"\n[NULLS] No nulls found. Nothing to drop.")

# ── DUPLICATE CHECK ──────────────────────────────────────────────────────────
n_dupes = df.duplicated().sum()
if n_dupes > 0:
    print(f"\n[DUPES] Found {n_dupes} duplicate rows — removing ...")
    df.drop_duplicates(inplace=True)
    print(f"        Shape after dedup : {df.shape}")
else:
    print(f"[DUPES] No duplicates found.")

# ── FEATURE / TARGET SPLIT VERIFICATION ──────────────────────────────────────
feature_cols = [c for c in df.columns if c != TARGET_COL]
print(f"\n[VERIFY] Feature columns ({len(feature_cols)}) :")
for col in feature_cols:
    print(f"  {col:<25}  dtype={str(df[col].dtype)}")

print(f"\n[VERIFY] Target column : '{TARGET_COL}'")
print(f"         Class distribution :")
counts = df[TARGET_COL].value_counts()
for cls, cnt in counts.items():
    print(f"           {cls:<25}  {cnt:>6}  ({cnt/len(df)*100:.1f}%)")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[SAVE] Written to : {OUTPUT_PATH}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLEANING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Input  shape  : {len(df)} rows × {len(df.columns)} columns  (after cleaning)")
print(f"  Features      : {len(feature_cols)}")
print(f"  Target        : {TARGET_COL}  ({df[TARGET_COL].nunique()} classes)")
print(f"  Nulls         : 0")
print(f"  Duplicates    : 0")
print(f"  Output file   : {OUTPUT_PATH}")
print("=" * 60)
print("\nNext step : run your preprocessing pipeline (uav_pipeline.py)")
print("  Remember to update pipeline config:")
print("    TARGET_COL       = 'label'")
print("    N_PCA_COMPONENTS = 16")
print("    CLASS_NAMES      = [")
print("        'Normal Traffic', 'Blackhole Attack', 'Wormhole Attack',")
print("        'Sybil Attack', 'Flooding Attack'")
print("    ]")