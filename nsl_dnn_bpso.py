"""
nsl_dnn_bpso.py
Drop-in integration of BPSO feature selection into the existing DNN pipeline.
Place this file next to feature_selection_bpso.py and your original nsl_dnn_updated.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from feature_selection_bpso import run_bpso   # <-- the only new import

np.random.seed(42)
tf.random.set_seed(42)

# Suppress Keras UserWarning about passing `input_shape`/`input_dim` to layers.
# This is a temporary suppression; prefer fixing the model definition later.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Do not pass an `input_shape`/`input_dim` argument to a layer.*",
    category=UserWarning,
)

# -------------------------
# 1. Load & Preprocess  (identical to the existing pipeline)
# -------------------------

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df  = pd.read_csv("KDDTest+.txt",  names=columns)

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label']  = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty',  axis=1, inplace=True)

categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()
for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col]  = encoder.transform(test_df[col])

X_train_full = train_df.drop('label', axis=1).values
y_train_full = train_df['label'].values
X_test       = test_df.drop('label', axis=1).values
y_test       = test_df['label'].values
feature_names = train_df.drop('label', axis=1).columns.tolist()

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled       = scaler.transform(X_test)

# -------------------------
# 2. Split a validation set for BPSO fitness evaluation
# BPSO needs a held-out set it has never seen — do NOT use X_test for this.
# -------------------------

X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full
)

# -------------------------
# 3. Run BPSO to get the best feature mask
# Tune n_particles / n_iter to trade speed vs. quality.
# For a quick run:  n_particles=10, n_iter=20
# For a proper run: n_particles=20, n_iter=50
# -------------------------

best_mask = run_bpso(
    X_train_scaled, y_train,
    X_val_scaled,   y_val,
    n_particles=10,
    n_iter=20,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    v_max=4.0
)

selected_indices = np.where(best_mask == 1)[0]
print("\nSelected features:", [feature_names[i] for i in selected_indices])

# -------------------------
# 4. Apply mask — this is the only change to the existing DNN pipeline
# -------------------------

X_train_bpso = X_train_scaled[:, selected_indices]
X_val_bpso   = X_val_scaled[:,   selected_indices]
X_test_bpso  = X_test_scaled[:,  selected_indices]

# -------------------------
# 5. Train the original DNN on the reduced feature set
# -------------------------

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Combine train + val back for final DNN training
X_final_train = np.vstack([X_train_bpso, X_val_bpso])
y_final_train = np.concatenate([y_train, y_val])

model = build_model(X_final_train.shape[1])
model.fit(
    X_final_train, y_final_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

y_pred = (model.predict(X_test_bpso) > 0.5).astype(int).flatten()

# -------------------------
# 6. Evaluation
# -------------------------

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n--- BPSO-Optimised DNN Results ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1        : {f1:.4f}")
print(f"Features used: {len(selected_indices)}/{X_test_scaled.shape[1]}")

# Per-class metrics (matching the style of nsl_dnn_bgwo.py)
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
    print(f"  {cls_name:6s} | Acc={cls_acc:.4f}  Prec={cls_prec:.4f}"
          f"  Rec={cls_rec:.4f}  F1={cls_f1:.4f}")
