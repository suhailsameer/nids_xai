"""
nsl_dnn_bgwo.py
===============
Full 5-step feature-selection pipeline for the NSL-KDD IDS benchmark:

  Step 1  Train a baseline DNN on the full feature set.
  Step 2  Compute SHAP importances (DeepExplainer → GradientExplainer → Kernel).
  Step 3  Pre-filter to top-k (or 95 % cumulative importance) features.
  Step 4  Run BGWO on the reduced candidate set.
  Step 5  Train the final DNN on the BGWO-selected features.
  (Bonus) LIME spot-check on misclassified test samples.

Dependencies
------------
    pip install shap lime tensorflow scikit-learn pandas numpy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from feature_selection import (
    build_baseline_dnn,
    compute_shap_importances,
    shap_prefilter,
    run_bgwo,
    lime_spot_check,
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

BGWO_N_WOLVES         = 20     # increase for better exploration (slower)
BGWO_N_ITER           = 50     # increase for more iterations (slower)

LIME_N_MISCLASSIFIED  = 5      # how many misclassifications to spot-check with LIME


# =============================================================================
# 1.  Load & preprocess  (identical to original pipeline)
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

categorical_cols = ["protocol_type", "service", "flag"]
encoder = LabelEncoder()
for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col]  = encoder.transform(test_df[col])

feature_names = list(train_df.drop("label", axis=1).columns)

X_train_full = train_df.drop("label", axis=1).values
y_train_full = train_df["label"].values
X_test       = test_df.drop("label", axis=1).values
y_test       = test_df["label"].values

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled       = scaler.transform(X_test)

# Validation split — never seen by BGWO or final DNN during feature search
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
)

y_pred_baseline = (baseline_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
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
    model       = baseline_model,
    X_background= X_train_scaled[bg_idx],
    X_explain   = X_val_scaled[exp_idx],
    method      = SHAP_METHOD,
    verbose     = True,
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

print("Candidate features for BGWO search:")
for i, fi in enumerate(candidate_indices, 1):
    print(f"  {i:2d}. {feature_names[fi]}")


# =============================================================================
# Step 4  BGWO on the reduced candidate set
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4  Running BGWO on SHAP-filtered candidates")
print("=" * 60)

best_mask = run_bgwo(
    X_train          = X_train_scaled,
    y_train          = y_train,
    X_val            = X_val_scaled,
    y_val            = y_val,
    n_wolves         = BGWO_N_WOLVES,
    n_iter           = BGWO_N_ITER,
    candidate_indices= candidate_indices,   # <-- the new argument
)

selected_indices = np.where(best_mask == 1)[0]
selected_names   = [feature_names[i] for i in selected_indices]
print(f"\nBGWO selected {len(selected_indices)}/{n_features_total} features:")
for name in selected_names:
    print(f"  • {name}")


# =============================================================================
# Step 5  Final DNN on BGWO-selected features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5  Training final DNN on BGWO-selected feature set")
print("=" * 60)

X_train_bgwo = X_train_scaled[:, selected_indices]
X_val_bgwo   = X_val_scaled[:,   selected_indices]
X_test_bgwo  = X_test_scaled[:,  selected_indices]

# Merge train + val for the final DNN (BGWO search is complete)
X_final_train = np.vstack([X_train_bgwo, X_val_bgwo])
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

y_pred_final = (final_model.predict(X_test_bgwo) > 0.5).astype(int).flatten()

print("\n--- BGWO-Optimised DNN Results ---")
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


# =============================================================================
# Bonus  LIME spot-check on misclassified samples
# =============================================================================

print("\n" + "=" * 60)
print("BONUS  LIME spot-check on misclassified test samples")
print("=" * 60)

misclassified = np.where(y_pred_final != y_test)[0]
print(f"Total misclassified: {len(misclassified)}")

if len(misclassified) > 0:
    check_indices = misclassified[:LIME_N_MISCLASSIFIED].tolist()
    selected_feature_names = [feature_names[i] for i in selected_indices]

    lime_spot_check(
        model            = final_model,
        X_background     = X_train_bgwo,           # background in the reduced space
        X_instances      = X_test_bgwo,
        feature_names    = selected_feature_names,
        instance_indices = check_indices,
        n_samples        = 1000,
        num_features     = 10,
        verbose          = True,
    )
else:
    print("No misclassifications found — perfect test set accuracy!")
