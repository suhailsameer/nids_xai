"""
nsl_dnn_bgwo_pytorch.py
PyTorch implementation of BGWO feature selection integrated with a DNN pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from feature_selection_pytorch import run_bgwo

np.random.seed(42)
torch.manual_seed(42)

# -------------------------
# 1. Load & Preprocess  (identical to your existing pipeline)
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
# 2. Split a validation set for BGWO fitness evaluation
# -------------------------

X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full
)

# -------------------------
# 3. Run BGWO to get the best feature mask
# -------------------------

best_mask = run_bgwo(
    X_train_scaled, y_train,
    X_val_scaled,   y_val,
    n_wolves=10,
    n_iter=20
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
# 5. Train your DNN on the reduced feature set using PyTorch
# -------------------------

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_bgwo, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor   = torch.tensor(X_val_bgwo, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor  = torch.tensor(X_test_bgwo, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Initialize model, loss, and optimizer
model = DNN(X_train_bgwo.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Testing
model.eval()
with torch.no_grad():
    y_pred = (model(X_test_tensor) > 0.5).int()

print("\n--- BGWO-Optimised DNN Results ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1        : {f1_score(y_test, y_pred):.4f}")
print(f"Features used: {len(selected_indices)}/{X_test_scaled.shape[1]}")