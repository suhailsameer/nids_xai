"""
nsl_dnn_bpso.py
===============
Full 5-step feature-selection pipeline for the NSL-KDD IDS benchmark
using Binary Particle Swarm Optimization (BPSO):

  Step 1  Train a baseline DNN on the full feature set.
  Step 2  Compute SHAP importances (DeepExplainer → GradientExplainer → Kernel).
  Step 3  Pre-filter to top-k (or 95 % cumulative importance) features.
  Step 4  Run BPSO on the reduced candidate set.
  Step 5  Train the final DNN on the BPSO-selected features.

Dependencies
------------
    pip install shap tensorflow scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xai_feature_selection_bpso import (
    build_baseline_dnn,
    compute_shap_importances,
    shap_prefilter,
    run_bpso,
)

np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# 0.  Configuration — edit these to tune speed vs quality
# =============================================================================

SHAP_BACKGROUND_SIZE  = 100    # rows sampled from training set for SHAP background
SHAP_EXPLAIN_SIZE     = 500    # rows explained to compute mean(|SHAP|)
SHAP_METHOD           = "auto" # 'auto' | 'deep' | 'gradient' | 'kernel'

PREFILTER_TOP_K       = 25     # set to None to use cumulative threshold instead
PREFILTER_CUMULATIVE  = None   # e.g. 0.95; only used when PREFILTER_TOP_K is None

BPSO_N_PARTICLES      = 10     # increase for better exploration (slower)
BPSO_N_ITER           = 10     # increase for more iterations (slower)


# =============================================================================
# 1.  Load & preprocess
# =============================================================================

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

train_df = pd.read_csv("KDDTrain+.txt", names=COLUMNS)
test_df  = pd.read_csv("KDDTest+.txt",  names=COLUMNS)

train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_df["label"]  = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

train_df.drop("difficulty", axis=1, inplace=True)
test_df.drop("difficulty",  axis=1, inplace=True)

# One-hot encoding for categorical columns (consistent across train/test)
categorical_cols = ["protocol_type", "service", "flag"]
combined_all = pd.concat([train_df, test_df], ignore_index=True)
combined_all = pd.get_dummies(combined_all, columns=categorical_cols)

# Split back into train and test
train_df = combined_all.iloc[: len(train_df)].copy()
test_df  = combined_all.iloc[len(train_df) :].copy()

feature_names = list(train_df.drop("label", axis=1).columns)

X_train_full = train_df.drop("label", axis=1).values
y_train_full = train_df["label"].values
X_test       = test_df.drop("label", axis=1).values
y_test       = test_df["label"].values

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled       = scaler.transform(X_test)

# Validation split — never seen by BPSO or final DNN during feature search
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full,
)

n_features_total = X_train_scaled.shape[1]


# =============================================================================
# Step 1  Baseline DNN — trained on the full feature set
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1  Training baseline DNN (full feature set)")
print("=" * 60)

baseline_model = build_baseline_dnn(n_features_total)
baseline_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_val_scaled, y_val),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ],
    verbose=1,
    class_weight={
        i: w for i, w in enumerate(
            compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        )
    },
)

# Tune probability threshold on the validation set to maximise F1
probs_val = baseline_model.predict(X_val_scaled).ravel()
prec_v, rec_v, thr_v = precision_recall_curve(y_val, probs_val)
f1s_v = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)
best_idx_v = int(np.nanargmax(f1s_v[:-1])) if len(f1s_v) > 1 else 0
best_threshold = thr_v[best_idx_v] if len(thr_v) > 0 else 0.5
print(f"[Baseline] Best val threshold={best_threshold:.3f}  (val F1={f1s_v[best_idx_v]:.4f})")

probs_test_baseline = baseline_model.predict(X_test_scaled).ravel()
y_pred_baseline     = (probs_test_baseline >= best_threshold).astype(int)

print("\n--- Baseline DNN Results ---")
print(classification_report(y_test, y_pred_baseline, target_names=["normal", "attack"]))


# =============================================================================
# Step 2  SHAP importances
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2  Computing SHAP importances")
print("=" * 60)

rng = np.random.default_rng(0)
bg_idx  = rng.choice(len(X_train_scaled), size=SHAP_BACKGROUND_SIZE, replace=False)
exp_idx = rng.choice(len(X_val_scaled),   size=SHAP_EXPLAIN_SIZE,    replace=False)

shap_importances = compute_shap_importances(
    model        = baseline_model,
    X_background = X_train_scaled[bg_idx],
    X_explain    = X_val_scaled[exp_idx],
    method       = SHAP_METHOD,
    verbose      = True,
)

# Print full ranked list
ranked = np.argsort(shap_importances)[::-1]
print("\nFull SHAP ranking (feature  →  mean|SHAP|):")
for rank, fi in enumerate(ranked, 1):
    print(f"  {rank:2d}. {feature_names[fi]:<45s} {shap_importances[fi]:.5f}")


# =============================================================================
# Step 3  SHAP pre-filter
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3  Pre-filtering features with SHAP")
print("=" * 60)

candidate_indices = shap_prefilter(
    importances          = shap_importances,
    top_k                = PREFILTER_TOP_K,
    cumulative_threshold = PREFILTER_CUMULATIVE,
    verbose              = True,
)

print("Candidate features for BPSO search:")
for i, fi in enumerate(candidate_indices, 1):
    print(f"  {i:2d}. {feature_names[fi]}")


# =============================================================================
# Step 4  BPSO on the reduced candidate set
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4  Running BPSO on SHAP-filtered candidates")
print("=" * 60)

best_mask = run_bpso(
    X_train           = X_train_scaled,
    y_train           = y_train,
    X_val             = X_val_scaled,
    y_val             = y_val,
    n_particles       = BPSO_N_PARTICLES,
    n_iter            = BPSO_N_ITER,
    w_max             = 0.9,
    w_min             = 0.4,
    c1                = 2.0,
    c2                = 2.0,
    v_max             = 4.0,
    candidate_indices = candidate_indices,
    fitness_epochs    = 15,
    fitness_batch_size= 256,
    verbose           = True,
)

selected_indices = np.where(best_mask == 1)[0]
selected_names   = [feature_names[i] for i in selected_indices]
print(f"\nBPSO selected {len(selected_indices)}/{n_features_total} features:")
for name in selected_names:
    print(f"  • {name}")


# =============================================================================
# Step 5  Final DNN on BPSO-selected features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5  Training final DNN on BPSO-selected feature set")
print("=" * 60)

X_train_bpso = X_train_scaled[:, selected_indices]
X_val_bpso   = X_val_scaled[:,   selected_indices]
X_test_bpso  = X_test_scaled[:,  selected_indices]

# Merge train + val for the final DNN (BPSO search is complete)
X_final_train = np.vstack([X_train_bpso, X_val_bpso])
y_final_train = np.concatenate([y_train, y_val])

final_model = build_baseline_dnn(len(selected_indices))
final_model.fit(
    X_final_train, y_final_train,
    epochs=100,
    batch_size=256,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ],
    verbose=1,
)

probs_test_final = final_model.predict(X_test_bpso).ravel()
y_pred_final     = (probs_test_final > 0.5).astype(int)


# =============================================================================
# Results & metrics
# =============================================================================

print("\n--- BPSO-Optimised DNN Results ---")
print(classification_report(y_test, y_pred_final, target_names=["normal", "attack"]))
print(f"Accuracy  : {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_final):.4f}")
print(f"F1        : {f1_score(y_test, y_pred_final):.4f}")
print(f"Features  : {len(selected_indices)}/{n_features_total}")

print("\n--- Improvement over baseline ---")
base_f1  = f1_score(y_test, y_pred_baseline)
final_f1 = f1_score(y_test, y_pred_final)
print(f"Baseline F1 : {base_f1:.4f}")
print(f"Final F1    : {final_f1:.4f}  (delta = {final_f1 - base_f1:+.4f})")

# Per-class breakdown
cm = confusion_matrix(y_test, y_pred_final)
print("\n--- Per-class breakdown ---")
TN, FP, FN, TP = cm.ravel()
for cls_name, tp, fp, fn, tn in [
    ("Normal", TN, FN, FP, TP),
    ("Attack", TP, FP, FN, TN),
]:
    cls_acc  = (tp + tn) / (tp + tn + fp + fn)
    cls_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cls_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cls_f1   = (2 * cls_prec * cls_rec / (cls_prec + cls_rec)
                if (cls_prec + cls_rec) > 0 else 0.0)
    print(
        f"  {cls_name:6s} | Acc={cls_acc:.4f}  Prec={cls_prec:.4f}"
        f"  Rec={cls_rec:.4f}  F1={cls_f1:.4f}"
    )


# =============================================================================
# Plot 1 — Precision-Recall curves: Baseline vs BPSO-Optimised
# =============================================================================

precision_b, recall_b, _ = precision_recall_curve(y_test, probs_test_baseline)
precision_f, recall_f, _ = precision_recall_curve(y_test, probs_test_final)
ap_b = average_precision_score(y_test, probs_test_baseline)
ap_f = average_precision_score(y_test, probs_test_final)

plt.figure(figsize=(8, 6))
plt.plot(recall_b, precision_b, label=f"Baseline (AP={ap_b:.3f})", lw=2)
plt.plot(recall_f, precision_f, label=f"BPSO Final (AP={ap_f:.3f})", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Baseline vs BPSO-Optimized")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curve_comparison_bpso.png", dpi=150)
plt.show()
print("Saved: pr_curve_comparison_bpso.png")


# =============================================================================
# Plot 2 — Confusion matrices side-by-side: Baseline vs BPSO-Optimised
# =============================================================================

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_final    = confusion_matrix(y_test, y_pred_final)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm_data, title in [
    (axes[0], cm_baseline, "Baseline DNN\n(full feature set)"),
    (axes[1], cm_final,    f"BPSO-Optimised DNN\n({len(selected_indices)} features)"),
]:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_data,
        display_labels=["Normal", "Attack"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)

fig.suptitle("Confusion Matrices: Baseline vs BPSO-Optimized", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrix_bpso.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrix_bpso.png")
