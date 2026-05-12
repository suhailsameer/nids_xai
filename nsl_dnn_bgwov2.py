"""
nsl_dnn_bgwo.py  (improved)
BGWO feature selection + optimised DNN pipeline for NSL-KDD.

Key improvements over v1:
  - Class imbalance: compute_class_weight passed to model.fit
  - Deeper DNN: 256 → 128 → 64 → 32, with BatchNormalization
  - Lower dropout (0.3 → 0.2) — previous rate was killing signal
  - EarlyStopping (patience=10) + ReduceLROnPlateau to avoid over/under-fitting
  - More epochs (100) — early stopping decides the actual stopping point
  - Decision threshold tuned on validation set for best F1 (not hard 0.5)
  - Confusion matrix + classification report printed for full diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from feature_selection_4 import run_bgwo

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# GPU check
# -------------------------
print("Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
print(f"{'GPUs detected: ' + str([g.name for g in gpus]) if gpus else 'No GPU — using CPU.'}")

# -------------------------
# 1. Load & Preprocess
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

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled       = scaler.transform(X_test)

# -------------------------
# 2. Train / Validation split for BGWO
# -------------------------
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full
)

# -------------------------
# 3. Run BGWO  (raised to n_wolves=30, n_iter=60 for a proper search)
# -------------------------
best_mask = run_bgwo(
    X_train_scaled, y_train,
    X_val_scaled,   y_val,
    n_wolves=30,
    n_iter=60
)

selected_indices = np.where(best_mask == 1)[0]
feature_names    = train_df.drop('label', axis=1).columns
print("\nSelected features:", list(feature_names[selected_indices]))

# -------------------------
# 4. Apply mask
# -------------------------
X_train_bgwo = X_train_scaled[:, selected_indices]
X_val_bgwo   = X_val_scaled[:,   selected_indices]
X_test_bgwo  = X_test_scaled[:,  selected_indices]

# -------------------------
# 5. Class weights  (fix imbalance in DNN training)
# -------------------------
classes = np.unique(y_train_full)
weights = compute_class_weight('balanced', classes=classes, y=y_train_full)
class_weight_dict = dict(zip(classes, weights))
print(f"\nClass weights: {class_weight_dict}")

# -------------------------
# 6. Improved DNN
#    256 → BN → 128 → BN → 64 → BN → 32 → 1
#    Dropout reduced to 0.2; BatchNorm stabilises training
# -------------------------
def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Combine train + val for final DNN training
X_final_train = np.vstack([X_train_bgwo, X_val_bgwo])
y_final_train = np.concatenate([y_train, y_val])

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,          # stop if no improvement for 10 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # halve LR on plateau
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

model = build_model(X_final_train.shape[1])
model.fit(
    X_final_train, y_final_train,
    epochs=100,               # early stopping will trigger before this
    batch_size=256,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# 7. Threshold tuning on validation set
#    Sweep thresholds 0.3–0.7, pick the one maximising F1
# -------------------------
val_probs = model.predict(X_val_bgwo).flatten()
best_thresh, best_val_f1 = 0.5, 0.0
for thresh in np.arange(0.3, 0.71, 0.01):
    preds = (val_probs > thresh).astype(int)
    score = f1_score(y_val, preds, zero_division=0)
    if score > best_val_f1:
        best_val_f1  = score
        best_thresh  = thresh

print(f"\nOptimal threshold: {best_thresh:.2f}  (val F1={best_val_f1:.4f})")

# -------------------------
# 8. Final evaluation on test set
# -------------------------
test_probs = model.predict(X_test_bgwo).flatten()
y_pred = (test_probs > best_thresh).astype(int)

print("\n--- BGWO-Optimised DNN Results (improved) ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1        : {f1_score(y_test, y_pred):.4f}")
print(f"Features used: {len(selected_indices)}/{X_test_scaled.shape[1]}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))