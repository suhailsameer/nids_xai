"""
feature_selection_bpso.py
Binary Particle Swarm Optimization (BPSO) for feature selection in IDS pipelines.
Fitness is evaluated using a lightweight DNN (Keras) on the masked feature subset.

Usage:
    from feature_selection_bpso import run_bpso
    best_mask = run_bpso(X_train, y_train, X_val, y_val)
    X_train_reduced = X_train[:, best_mask == 1]
    X_test_reduced  = X_test[:, best_mask == 1]
"""

import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Suppress Keras UserWarning about passing `input_shape`/`input_dim` to layers.
# This is a temporary suppression; prefer fixing the model definition later.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Do not pass an `input_shape`/`input_dim` argument to a layer.*",
    category=UserWarning,
)
# ---------------------------------------------------------------------------
# Fitness DNN builder — intentionally smaller than the final model.
# Fewer epochs + early stopping keep each evaluation fast.
# ---------------------------------------------------------------------------

def _build_fitness_dnn(input_dim: int) -> Sequential:
    """Small DNN used only inside BPSO fitness evaluation."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


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

    X_tr = X_train[:, selected]
    X_v  = X_val[:, selected]

    model = _build_fitness_dnn(len(selected))

    early_stop = EarlyStopping(monitor='val_loss', patience=2,
                               restore_best_weights=True, verbose=0)

    model.fit(
        X_tr, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_v, y_val),
        callbacks=[early_stop],
        verbose=0          # silence per-epoch output inside BPSO
    )

    y_pred = (model.predict(X_v, verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_val, y_pred)

    # Free GPU/CPU memory from this temporary model
    del model
    tf.keras.backend.clear_session()

    penalty = len(selected) / X_train.shape[1]
    return 0.9 * acc - 0.1 * penalty


# ---------------------------------------------------------------------------
# Sigmoid transfer function  (continuous velocity → probability of bit = 1)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    # Clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Core BPSO
# ---------------------------------------------------------------------------

def run_bpso(X_train, y_train, X_val, y_val,
             n_particles: int = 10,
             n_iter:      int = 20,
             w_max:       float = 0.9,
             w_min:       float = 0.4,
             c1:          float = 2.0,
             c2:          float = 2.0,
             v_max:       float = 4.0) -> np.ndarray:
    """
    Binary Particle Swarm Optimization for feature selection.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data (used only inside fitness)
    n_particles      : swarm size (more = better exploration, slower)
    n_iter           : number of iterations
    w_max, w_min     : inertia weight linearly decayed from w_max → w_min
                       (high inertia early → exploration; low inertia later → exploitation)
    c1               : cognitive coefficient — personal best pull strength
    c2               : social coefficient   — global best pull strength
    v_max            : velocity clamp; keeps sigmoid probabilities away from 0/1 extremes

    Returns
    -------
    best_mask : np.ndarray of shape (n_features,) with values 0 or 1
    """
    n_features = X_train.shape[1]
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Initialise swarm
    #   positions  : binary matrix  (n_particles × n_features)
    #   velocities : continuous matrix (n_particles × n_features), small init
    # -----------------------------------------------------------------------
    positions  = rng.integers(0, 2, size=(n_particles, n_features)).astype(float)
    velocities = rng.uniform(-v_max / 2.0, v_max / 2.0,
                             size=(n_particles, n_features))

    # Evaluate initial swarm
    fitness = np.array([
        _fitness(positions[i].astype(int), X_train, y_train, X_val, y_val)
        for i in range(n_particles)
    ])

    # Personal bests — each particle remembers its own best position
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    # Global best — best position ever seen across the whole swarm
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    best_mask = positions[gbest_idx].astype(int)
    best_fit  = gbest_fit

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for iteration in range(n_iter):
        # Linearly decay inertia: promotes exploration early, exploitation late
        w = w_max - (w_max - w_min) * (iteration / n_iter)

        for i in range(n_particles):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)

            # Standard PSO velocity update equation
            cognitive      = c1 * r1 * (pbest_pos[i] - positions[i])
            social         = c2 * r2 * (gbest_pos    - positions[i])
            velocities[i]  = w * velocities[i] + cognitive + social

            # Clamp velocity to keep sigmoid in a useful range
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Sigmoid transfer: velocity → probability of feature being selected
            prob          = _sigmoid(velocities[i])
            positions[i]  = (rng.random(n_features) < prob).astype(float)

        # Evaluate updated swarm
        fitness = np.array([
            _fitness(positions[i].astype(int), X_train, y_train, X_val, y_val)
            for i in range(n_particles)
        ])

        # Update personal bests (vectorised comparison)
        improved           = fitness > pbest_fit
        pbest_pos[improved] = positions[improved].copy()
        pbest_fit[improved] = fitness[improved]

        # Update global best
        gbest_idx = np.argmax(pbest_fit)
        if pbest_fit[gbest_idx] > gbest_fit:
            gbest_fit = pbest_fit[gbest_idx]
            gbest_pos = pbest_pos[gbest_idx].copy()

        if gbest_fit > best_fit:
            best_fit  = gbest_fit
            best_mask = gbest_pos.astype(int)

        n_sel = int(best_mask.sum())
        print(f"Iter {iteration + 1:3d}/{n_iter} | "
              f"best_fitness={best_fit:.4f} | selected_features={n_sel}")

    print(f"\nBPSO complete. Selected {int(best_mask.sum())}/{n_features} features.")
    return best_mask
