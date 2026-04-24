"""
feature_selection_pytorch.py
Binary Grey Wolf Optimizer (BGWO) for feature selection in IDS pipelines using PyTorch.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Fitness DNN builder — intentionally smaller than your final model.
# Fewer epochs + early stopping keep each evaluation fast.
# ---------------------------------------------------------------------------

class FitnessDNN(nn.Module):
    def __init__(self, input_dim):
        super(FitnessDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# ---------------------------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------------------------

def _fitness(mask, X_train, y_train, X_val, y_val,
             epochs: int = 10, batch_size: int = 256) -> float:
    """
    Evaluate a binary feature mask using a small DNN.

    Returns a scalar score to MAXIMISE:
        fitness = 0.9 * accuracy - 0.1 * (selected / total)

    A mask with no selected features gets a score of 0.
    """
    selected = np.where(mask == 1)[0]
    if len(selected) == 0:
        return 0.0

    X_tr = torch.tensor(X_train[:, selected], dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v  = torch.tensor(X_val[:, selected], dtype=torch.float32)
    y_v  = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = FitnessDNN(len(selected))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_v)
        val_loss = criterion(val_outputs, y_v)
        y_pred = (val_outputs > 0.5).int()
        acc = accuracy_score(y_v.numpy(), y_pred.numpy())

    penalty = len(selected) / X_train.shape[1]
    return 0.9 * acc - 0.1 * penalty

# ---------------------------------------------------------------------------
# Sigmoid transfer function  (continuous → probability of bit = 1)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    # Clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ---------------------------------------------------------------------------
# Core BGWO
# ---------------------------------------------------------------------------

def run_bgwo(X_train, y_train, X_val, y_val,
             n_wolves: int = 10,
             n_iter:   int = 20) -> np.ndarray:
    """
    Binary Grey Wolf Optimizer for feature selection.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data (used only inside fitness)
    n_wolves         : population size (more = better exploration, slower)
    n_iter           : number of iterations

    Returns
    -------
    best_mask : np.ndarray of shape (n_features,) with values 0 or 1
    """
    n_features = X_train.shape[1]
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Initialise population: each wolf is a continuous position vector in
    # [-4, 4].  We binarise via sigmoid for fitness evaluation.
    # -----------------------------------------------------------------------
    positions = rng.uniform(-4, 4, size=(n_wolves, n_features))  # continuous

    def _binarise(pos):
        prob = _sigmoid(pos)
        return (rng.random(n_features) < prob).astype(int)

    # Evaluate initial population
    fitness = np.array([
        _fitness(_binarise(positions[i]), X_train, y_train, X_val, y_val)
        for i in range(n_wolves)
    ])

    # Identify alpha (best), beta (2nd), delta (3rd)
    sorted_idx = np.argsort(fitness)[::-1]
    alpha_pos  = positions[sorted_idx[0]].copy()
    alpha_fit  = fitness[sorted_idx[0]]
    beta_pos   = positions[sorted_idx[1]].copy()
    delta_pos  = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

    best_mask  = _binarise(alpha_pos)
    best_fit   = alpha_fit

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for iteration in range(n_iter):
        # a decreases linearly from 2 → 0 over iterations
        a = 2.0 - 2.0 * (iteration / n_iter)

        for i in range(n_wolves):
            new_pos = np.empty(n_features)

            for j in range(n_features):
                # --- alpha contribution ---
                r1, r2 = rng.random(), rng.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # --- beta contribution ---
                r1, r2 = rng.random(), rng.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                # --- delta contribution ---
                r1, r2 = rng.random(), rng.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                new_pos[j] = (X1 + X2 + X3) / 3.0

            positions[i] = new_pos

        # Evaluate updated population
        fitness = np.array([
            _fitness(_binarise(positions[i]), X_train, y_train, X_val, y_val)
            for i in range(n_wolves)
        ])

        # Update alpha / beta / delta
        sorted_idx = np.argsort(fitness)[::-1]
        alpha_pos  = positions[sorted_idx[0]].copy()
        alpha_fit  = fitness[sorted_idx[0]]
        beta_pos   = positions[sorted_idx[1]].copy()
        delta_pos  = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

        if alpha_fit > best_fit:
            best_fit  = alpha_fit
            best_mask = _binarise(alpha_pos)

        n_sel = int(best_mask.sum())
        print(f"Iter {iteration + 1:3d}/{n_iter} | "
              f"best_fitness={best_fit:.4f} | selected_features={n_sel}")

    print(f"\nBGWO complete. Selected {int(best_mask.sum())}/{n_features} features.")
    return best_mask