"""
feature_selection_bpso.py
=========================
Binary Particle Swarm Optimization (BPSO) for feature selection in IDS pipelines,
augmented with:
  1. SHAP-based pre-filtering  – reduces the search space before BPSO runs.
  2. BPSO on the reduced candidate set  – faster, still meta-heuristic.

Public API
----------
    from feature_selection_bpso import (
        build_baseline_dnn,          # shared DNN builder
        compute_shap_importances,    # step 2
        shap_prefilter,              # step 3
        run_bpso,                    # step 4
    )
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


# ============================================================================
# 0.  Shared DNN builders
# ============================================================================

def _build_fitness_dnn(input_dim: int) -> Sequential:
    """Lightweight DNN used only inside BPSO fitness evaluation."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_dnn(input_dim: int) -> Sequential:
    """
    Full-size DNN identical to the one used in the main pipeline.
    Exported so the pipeline can import a single shared definition,
    and so SHAP can explain a properly-trained model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================================
# 1.  SHAP importance computation
# ============================================================================

def compute_shap_importances(
    model: Sequential,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    method: str = "auto",
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute mean(|SHAP|) per feature for a trained Keras binary classifier.

    Parameters
    ----------
    model        : trained Keras model
    X_background : background sample for SHAP (100–200 rows recommended)
    X_explain    : rows to explain (300–1000 rows recommended)
    method       : 'deep' | 'gradient' | 'kernel' | 'auto'
                   'auto' tries DeepExplainer → GradientExplainer → KernelExplainer
    verbose      : print which explainer was selected

    Returns
    -------
    importances : np.ndarray of shape (n_features,)
                  mean absolute SHAP value per feature, averaged over X_explain
    """
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP-based pre-filtering.\n"
            "Install with:  pip install shap"
        ) from exc

    import shap  # local re-import so type checkers are happy

    shap_values = None

    def _try_deep():
        try:
            explainer = shap.DeepExplainer(model, X_background)
            sv = explainer.shap_values(X_explain)
            return sv[0] if isinstance(sv, list) else sv
        except Exception:
            return None

    def _try_gradient():
        try:
            explainer = shap.GradientExplainer(model, X_background)
            sv = explainer.shap_values(X_explain)
            return sv[0] if isinstance(sv, list) else sv
        except Exception:
            return None

    def _try_kernel():
        def predict_fn(x):
            return model.predict(x, verbose=0).flatten()
        explainer = shap.KernelExplainer(predict_fn, X_background)
        sv = explainer.shap_values(X_explain, nsamples=100, verbose=0)
        return sv

    if method == "auto":
        for label, fn in [("DeepExplainer", _try_deep),
                          ("GradientExplainer", _try_gradient),
                          ("KernelExplainer", _try_kernel)]:
            shap_values = fn()
            if shap_values is not None:
                if verbose:
                    print(f"[SHAP] Using {label}")
                break
    elif method == "deep":
        shap_values = _try_deep()
        if shap_values is None:
            raise RuntimeError("DeepExplainer failed; try method='auto'.")
    elif method == "gradient":
        shap_values = _try_gradient()
        if shap_values is None:
            raise RuntimeError("GradientExplainer failed; try method='auto'.")
    elif method == "kernel":
        shap_values = _try_kernel()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose auto/deep/gradient/kernel.")

    if shap_values is None:
        raise RuntimeError("All SHAP explainers failed.")

    arr = np.asarray(shap_values)

    # Handle common returned shapes from different explainers:
    # - (n_samples, n_features)
    # - (1, n_samples, n_features)
    # - (n_samples, n_features, 1)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)

    if arr.ndim != 2:
        try:
            arr = arr.reshape(arr.shape[0], -1)
        except Exception:
            raise RuntimeError(f"Unexpected shap_values shape: {np.asarray(shap_values).shape}")

    if arr.shape[0] != X_explain.shape[0] and arr.shape[1] == X_explain.shape[0]:
        arr = arr.T

    if arr.shape[0] != X_explain.shape[0]:
        raise RuntimeError(
            f"SHAP returned unexpected shape {arr.shape} vs X_explain {X_explain.shape}"
        )

    importances = np.mean(np.abs(arr), axis=0)

    if verbose:
        top5 = np.argsort(importances)[::-1][:5]
        print(f"[SHAP] Top-5 feature indices by importance: {top5.tolist()}")
        print(f"[SHAP] Their importances: {importances[top5].round(5).tolist()}")

    return importances


# ============================================================================
# 2.  SHAP pre-filtering
# ============================================================================

def shap_prefilter(
    importances: np.ndarray,
    top_k: int | None = 25,
    cumulative_threshold: float | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Return a sorted index array selecting the most important features.

    Exactly one of `top_k` or `cumulative_threshold` should be set.

    Parameters
    ----------
    importances          : output of compute_shap_importances()
    top_k                : keep the top-k features by mean(|SHAP|)
    cumulative_threshold : keep features whose cumulative importance
                           (sorted desc) reaches this fraction, e.g. 0.95
    verbose              : print how many features were retained

    Returns
    -------
    selected_indices : np.ndarray of int, shape (k,)
    """
    if top_k is None and cumulative_threshold is None:
        raise ValueError("Provide either top_k or cumulative_threshold.")
    if top_k is not None and cumulative_threshold is not None:
        raise ValueError("Provide only one of top_k or cumulative_threshold.")

    sorted_idx = np.argsort(importances)[::-1]  # descending

    if top_k is not None:
        selected = sorted_idx[:top_k]
    else:
        cumulative = np.cumsum(importances[sorted_idx]) / importances.sum()
        cutoff = np.searchsorted(cumulative, cumulative_threshold) + 1
        selected = sorted_idx[:cutoff]

    if verbose:
        criterion = f"top_k={top_k}" if top_k else f"cumulative >= {cumulative_threshold}"
        print(
            f"[SHAP pre-filter] Kept {len(selected)}/{len(importances)} features "
            f"(criterion: {criterion})"
        )

    return np.sort(selected)  # return in original column order


# ============================================================================
# 3.  Sigmoid transfer + fitness function
# ============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _fitness(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 256,
    sample_frac: float = 1.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate a binary feature mask using a small DNN.

    Returns a scalar score to MAXIMISE:
        fitness = 0.95 * accuracy - 0.05 * (selected / total)

    A mask with no selected features gets a score of 0.
    """
    selected = np.where(mask == 1)[0]
    if len(selected) == 0:
        return 0.0

    if rng is None:
        rng = np.random.default_rng()

    if sample_frac < 1.0:
        n_train = X_train.shape[0]
        n_sample = max(10, int(n_train * sample_frac))
        idx = rng.choice(n_train, size=n_sample, replace=False)
        X_train_s = X_train[idx][:, selected]
        y_train_s = y_train[idx]
        n_val = X_val.shape[0]
        n_val_sample = max(10, int(n_val * min(1.0, sample_frac)))
        idxv = rng.choice(n_val, size=n_val_sample, replace=False)
        X_val_s = X_val[idxv][:, selected]
        y_val_s = y_val[idxv]
    else:
        X_train_s = X_train[:, selected]
        y_train_s = y_train
        X_val_s = X_val[:, selected]
        y_val_s = y_val

    if verbose:
        print(
            f"[fitness] selected={len(selected)} features | "
            f"train_samples={len(y_train_s)} | val_samples={len(y_val_s)} | "
            f"epochs={epochs}"
        )

    model = _build_fitness_dnn(len(selected))
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=0
    )

    model.fit(
        X_train_s, y_train_s,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_s, y_val_s),
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred = (model.predict(X_val[:, selected], verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_val, y_pred)

    del model
    tf.keras.backend.clear_session()

    penalty = len(selected) / X_train.shape[1]
    return 0.95 * acc - 0.05 * penalty


# ============================================================================
# 4.  Core BPSO – now operates on the SHAP-filtered candidate set
# ============================================================================

def run_bpso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_particles: int = 10,
    n_iter: int = 20,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    v_max: float = 4.0,
    candidate_indices: np.ndarray | None = None,
    fitness_epochs: int = 15,
    fitness_batch_size: int = 256,
    fitness_sample_frac: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Binary Particle Swarm Optimization for feature selection.

    Parameters
    ----------
    X_train, y_train  : training data (full feature set, pre-scaled)
    X_val,   y_val    : validation data (used only inside fitness)
    n_particles       : swarm size (more = better exploration, slower)
    n_iter            : number of iterations
    w_max, w_min      : inertia weight linearly decayed w_max → w_min
    c1                : cognitive coefficient — personal best pull strength
    c2                : social coefficient   — global best pull strength
    v_max             : velocity clamp; keeps sigmoid probabilities away from 0/1 extremes
    candidate_indices : output of shap_prefilter().
                        If provided, BPSO searches only among these columns.
                        The returned mask is still aligned to the FULL feature
                        space so downstream slicing works transparently.
    fitness_epochs    : epochs per fitness DNN evaluation
    fitness_batch_size: batch size per fitness DNN evaluation
    fitness_sample_frac: fraction of training data to use per fitness eval (speed vs quality)

    Returns
    -------
    best_mask : np.ndarray of shape (n_features_full,) with values 0 or 1
    """
    n_features_full = X_train.shape[1]

    # Work in the reduced candidate space if provided
    if candidate_indices is not None:
        X_tr = X_train[:, candidate_indices]
        X_v  = X_val[:,   candidate_indices]
        n_features = len(candidate_indices)
        if verbose:
            print(
                f"[BPSO] Searching over {n_features}/{n_features_full} "
                "SHAP-filtered candidates."
            )
    else:
        X_tr = X_train
        X_v  = X_val
        n_features = n_features_full

    if verbose:
        expected_evals = n_particles * n_iter
        print(
            f"[BPSO] Starting BPSO: n_particles={n_particles}, n_iter={n_iter}, "
            f"fitness_epochs={fitness_epochs}, fitness_batch_size={fitness_batch_size}, "
            f"fitness_sample_frac={fitness_sample_frac}, "
            f"expected_evaluations~={expected_evals}"
        )

    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Initialise swarm
    #   positions  : binary matrix  (n_particles × n_features)
    #   velocities : continuous matrix (n_particles × n_features), small init
    # -----------------------------------------------------------------------
    positions  = rng.integers(0, 2, size=(n_particles, n_features)).astype(float)
    velocities = rng.uniform(-v_max / 2.0, v_max / 2.0, size=(n_particles, n_features))

    # Evaluate initial swarm
    fitness = np.array([
        _fitness(
            positions[i].astype(int), X_tr, y_train, X_v, y_val,
            epochs=fitness_epochs, batch_size=fitness_batch_size,
            sample_frac=fitness_sample_frac, rng=rng, verbose=verbose,
        )
        for i in range(n_particles)
    ])

    # Personal bests — each particle remembers its own best position
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    # Global best — best position ever seen across the whole swarm
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    best_local_mask = positions[gbest_idx].astype(int)
    best_fit        = gbest_fit

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for iteration in range(n_iter):
        # Linearly decay inertia: exploration early, exploitation late
        w = w_max - (w_max - w_min) * (iteration / n_iter)

        if verbose:
            print(f"[BPSO] Iter {iteration + 1}/{n_iter} starting...")

        for i in range(n_particles):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)

            cognitive     = c1 * r1 * (pbest_pos[i] - positions[i])
            social        = c2 * r2 * (gbest_pos    - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity to keep sigmoid in a useful range
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Sigmoid transfer: velocity → probability of feature being selected
            prob         = _sigmoid(velocities[i])
            positions[i] = (rng.random(n_features) < prob).astype(float)

        # Evaluate updated swarm
        fitness = np.array([
            _fitness(
                positions[i].astype(int), X_tr, y_train, X_v, y_val,
                epochs=fitness_epochs, batch_size=fitness_batch_size,
                sample_frac=fitness_sample_frac, rng=rng, verbose=verbose,
            )
            for i in range(n_particles)
        ])

        # Update personal bests
        improved            = fitness > pbest_fit
        pbest_pos[improved] = positions[improved].copy()
        pbest_fit[improved] = fitness[improved]

        # Update global best
        gbest_idx = np.argmax(pbest_fit)
        if pbest_fit[gbest_idx] > gbest_fit:
            gbest_fit = pbest_fit[gbest_idx]
            gbest_pos = pbest_pos[gbest_idx].copy()

        if gbest_fit > best_fit:
            best_fit        = gbest_fit
            best_local_mask = gbest_pos.astype(int)

        n_sel = int(best_local_mask.sum())
        if verbose:
            print(
                f"Iter {iteration + 1:3d}/{n_iter} | "
                f"best_fitness={best_fit:.4f} | selected_features={n_sel}"
            )

    # Map local mask back to full feature space
    if candidate_indices is not None:
        best_mask_full = np.zeros(n_features_full, dtype=int)
        best_mask_full[candidate_indices[best_local_mask == 1]] = 1
    else:
        best_mask_full = best_local_mask

    if verbose:
        print(
            f"\nBPSO complete. Selected {int(best_mask_full.sum())}/{n_features_full} features."
        )
    return best_mask_full
