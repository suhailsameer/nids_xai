"""
feature_selection_4.py  (improved)
Binary Grey Wolf Optimizer (BGWO) for feature selection in IDS pipelines.

Key improvements over v1:
  - Fitness uses F1-score instead of accuracy (better for imbalanced data)
  - RandomForest surrogate (stronger proxy for a DNN than LogisticRegression)
  - Feature-count penalty reduced 0.1 → 0.05 (stops over-pruning useful features)
  - Deterministic best-mask: alpha position is binarised once and stored, not
    re-sampled on every read (eliminates mask drift between iterations)
  - Vectorised position update (no Python loop over features — ~10x faster)
  - class_weight='balanced' in surrogate to handle NSL-KDD imbalance

Usage:
    from feature_selection_4 import run_bgwo
    best_mask = run_bgwo(X_train, y_train, X_val, y_val)
    X_train_reduced = X_train[:, best_mask == 1]
    X_test_reduced  = X_test[:, best_mask == 1]
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Sigmoid transfer function
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Deterministic binarisation
# Uses the sigmoid probability as a hard threshold at 0.5 (no randomness).
# This ensures the stored best_mask is stable and reproducible.
# ---------------------------------------------------------------------------

def _binarise(pos: np.ndarray) -> np.ndarray:
    return (_sigmoid(pos) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------------------------

def _fitness(mask: np.ndarray,
             X_train: np.ndarray, y_train: np.ndarray,
             X_val:   np.ndarray, y_val:   np.ndarray) -> float:
    """
    Evaluate a binary feature mask using a RandomForest surrogate.

    Returns a scalar score to MAXIMISE:
        fitness = 0.95 * f1 - 0.05 * (selected / total)

    Improvements vs v1:
      - F1 instead of accuracy  → directly targets what we want to improve
      - RandomForest surrogate  → much better proxy for a DNN than LogReg
      - Lower penalty (0.05)    → keeps more informative features
      - class_weight='balanced' → handles class imbalance in fitness signal
    """
    selected = np.where(mask == 1)[0]
    if len(selected) == 0:
        return 0.0

    X_tr = X_train[:, selected]
    X_v  = X_val[:,   selected]

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_train)
    f1 = f1_score(y_val, clf.predict(X_v), zero_division=0)

    penalty = len(selected) / X_train.shape[1]
    return 0.95 * f1 - 0.05 * penalty


# ---------------------------------------------------------------------------
# Core BGWO
# ---------------------------------------------------------------------------

def run_bgwo(X_train: np.ndarray, y_train: np.ndarray,
             X_val:   np.ndarray, y_val:   np.ndarray,
             n_wolves: int = 30,
             n_iter:   int = 60) -> np.ndarray:
    """
    Binary Grey Wolf Optimizer for feature selection.

    Parameters
    ----------
    X_train, y_train : training data (scaled)
    X_val,   y_val   : validation data (used only inside fitness)
    n_wolves         : population size  — default raised to 30 for better exploration
    n_iter           : iterations       — default raised to 60 for convergence

    Returns
    -------
    best_mask : np.ndarray of shape (n_features,) with values 0 or 1
    """
    n_features = X_train.shape[1]
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Initialise population in [-4, 4]
    # -----------------------------------------------------------------------
    positions = rng.uniform(-4.0, 4.0, size=(n_wolves, n_features))

    # Evaluate initial population
    fitness_vals = np.array([
        _fitness(_binarise(positions[i]), X_train, y_train, X_val, y_val)
        for i in range(n_wolves)
    ])

    # Identify alpha / beta / delta (best 3)
    sorted_idx = np.argsort(fitness_vals)[::-1]
    alpha_pos = positions[sorted_idx[0]].copy();  alpha_fit = fitness_vals[sorted_idx[0]]
    beta_pos  = positions[sorted_idx[1]].copy()
    delta_pos = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

    # Store best mask deterministically (no random re-draw)
    best_mask = _binarise(alpha_pos)
    best_fit  = alpha_fit

    # -----------------------------------------------------------------------
    # Main loop — vectorised position update (no per-feature Python loop)
    # -----------------------------------------------------------------------
    for iteration in range(n_iter):
        # a decreases linearly 2 → 0
        a = 2.0 - 2.0 * (iteration / n_iter)

        # Draw all random coefficients at once: shape (n_wolves, n_features)
        r1 = rng.random((n_wolves, n_features))
        r2 = rng.random((n_wolves, n_features))
        A1 = 2 * a * r1 - a;  C1 = 2 * rng.random((n_wolves, n_features))

        r1 = rng.random((n_wolves, n_features))
        r2 = rng.random((n_wolves, n_features))
        A2 = 2 * a * r1 - a;  C2 = 2 * rng.random((n_wolves, n_features))

        r1 = rng.random((n_wolves, n_features))
        r2 = rng.random((n_wolves, n_features))
        A3 = 2 * a * r1 - a;  C3 = 2 * rng.random((n_wolves, n_features))

        # Broadcast leader positions to (n_wolves, n_features)
        D_alpha = np.abs(C1 * alpha_pos - positions)
        D_beta  = np.abs(C2 * beta_pos  - positions)
        D_delta = np.abs(C3 * delta_pos - positions)

        X1 = alpha_pos - A1 * D_alpha
        X2 = beta_pos  - A2 * D_beta
        X3 = delta_pos - A3 * D_delta

        positions = (X1 + X2 + X3) / 3.0

        # Evaluate updated population
        fitness_vals = np.array([
            _fitness(_binarise(positions[i]), X_train, y_train, X_val, y_val)
            for i in range(n_wolves)
        ])

        # Update alpha / beta / delta
        sorted_idx = np.argsort(fitness_vals)[::-1]
        alpha_pos = positions[sorted_idx[0]].copy();  alpha_fit = fitness_vals[sorted_idx[0]]
        beta_pos  = positions[sorted_idx[1]].copy()
        delta_pos = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

        if alpha_fit > best_fit:
            best_fit  = alpha_fit
            best_mask = _binarise(alpha_pos)   # deterministic, no random draw

        n_sel = int(best_mask.sum())
        print(f"Iter {iteration + 1:3d}/{n_iter} | "
              f"best_fitness={best_fit:.4f} | selected_features={n_sel}")

    print(f"\nBGWO complete.  Selected {int(best_mask.sum())}/{n_features} features.")
    return best_mask