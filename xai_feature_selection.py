"""
feature_selection.py
====================
Binary Grey Wolf Optimizer (BGWO) for feature selection in IDS pipelines,
now augmented with:
  1. SHAP-based pre-filtering  – reduces the search space before BGWO runs.
  2. BGWO on the reduced candidate set  – faster, still meta-heuristic.
  3. LIME spot-checking utilities  – explain individual predictions.

Public API
----------
    from feature_selection import (
        compute_shap_importances,   # step 2
        shap_prefilter,             # step 3
        run_bgwo,                   # step 4  (same signature as before)
        lime_spot_check,            # step 5
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
    """Lightweight DNN used only inside BGWO fitness evaluation."""
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
    Full-size DNN identical to the one in nsl_dnn_bgwo.py.
    Used here so SHAP can explain a properly-trained model.
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
            # DeepExplainer returns list[array] for multi-output; take index 0
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
        # KernelExplainer needs a predict function returning 1-D probabilities
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

    # Normalise shap_values to a 2-D array of shape (n_samples, n_features)
    arr = np.asarray(shap_values)

    # Handle common returned shapes from different explainers:
    # - (n_samples, n_features)
    # - (1, n_samples, n_features)
    # - (n_outputs, n_samples, n_features) -> we already selected index 0 for multi-output above
    # - (n_samples, n_features, 1)
    if arr.ndim == 3:
        # If first dim is 1, unwrap it
        if arr.shape[0] == 1:
            arr = arr[0]
        # If last dim is singleton, squeeze it
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)

    if arr.ndim != 2:
        # As a last resort try to reshape to (n_samples, n_features)
        try:
            arr = arr.reshape(arr.shape[0], -1)
        except Exception:
            raise RuntimeError(f"Unexpected shap_values shape: {np.asarray(shap_values).shape}")

    # Now arr should be (n_samples, n_features). If it's transposed, fix using X_explain's length
    if arr.shape[0] != X_explain.shape[0] and arr.shape[1] == X_explain.shape[0]:
        arr = arr.T

    if arr.shape[0] != X_explain.shape[0]:
        # Give a helpful error rather than failing silently
        raise RuntimeError(f"SHAP returned unexpected shape {arr.shape} vs X_explain {X_explain.shape}")

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
    Return a boolean index array selecting the most important features.

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
        print(f"[SHAP pre-filter] Kept {len(selected)}/{len(importances)} features "
              f"(criterion: {'top_k=' + str(top_k) if top_k else 'cumulative >= ' + str(cumulative_threshold)})")

    return np.sort(selected)  # return in original column order


# ============================================================================
# 3.  Fitness function (unchanged logic, minor refactor)
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
    Evaluate a binary feature mask with a small DNN.

    Returns a scalar to MAXIMISE:
        fitness = 0.95 * accuracy - 0.05 * (selected / total)
    """
    selected = np.where(mask == 1)[0]
    if len(selected) == 0:
        return 0.0
    model = _build_fitness_dnn(len(selected))
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=0
    )
    # Optionally subsample the training (and validation) sets to speed up fitness evals
    if sample_frac is None:
        sample_frac = 1.0
    if rng is None:
        rng = np.random.default_rng()

    if sample_frac < 1.0:
        n_train = X_train.shape[0]
        n_sample = max(10, int(n_train * sample_frac))
        idx = rng.choice(n_train, size=n_sample, replace=False)
        X_train_s = X_train[idx][:, selected]
        y_train_s = y_train[idx]
        # keep validation small too to speed up
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
        print(f"[fitness] selected={len(selected)} features | train_samples={len(y_train_s)} | val_samples={len(y_val_s)} | epochs={epochs}")

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
# 4.  BGWO – now operates on the SHAP-filtered candidate set
# ============================================================================

def run_bgwo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_wolves: int = 10,
    n_iter: int = 20,
    candidate_indices: np.ndarray | None = None,
    fitness_epochs: int = 15,
    fitness_batch_size: int = 256,
    fitness_sample_frac: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Binary Grey Wolf Optimizer for feature selection.

    Parameters
    ----------
    X_train, y_train   : training data (full feature set, pre-scaled)
    X_val,   y_val     : validation data
    n_wolves           : population size
    n_iter             : iterations
    candidate_indices  : output of shap_prefilter().
                         If provided, BGWO searches only among these columns.
                         The returned mask is still aligned to the FULL feature
                         space so downstream slicing works transparently.

    Returns
    -------
    best_mask : np.ndarray of shape (n_features_full,) with values 0 | 1
    """
    n_features_full = X_train.shape[1]

    # Work in the reduced candidate space if provided
    if candidate_indices is not None:
        X_tr = X_train[:, candidate_indices]
        X_v  = X_val[:,   candidate_indices]
        n_features = len(candidate_indices)
        if verbose:
            print(f"[BGWO] Searching over {n_features}/{n_features_full} SHAP-filtered candidates.")
    else:
        X_tr = X_train
        X_v  = X_val
        n_features = n_features_full

    if verbose:
        expected_evals = n_wolves * n_iter
        print(f"[BGWO] Starting BGWO: n_wolves={n_wolves}, n_iter={n_iter}, "
              f"fitness_epochs={fitness_epochs}, fitness_batch_size={fitness_batch_size}, "
              f"fitness_sample_frac={fitness_sample_frac}, expected_evaluations~={expected_evals}")

    rng = np.random.default_rng(42)

    # Initialise continuous positions in [-4, 4]
    positions = rng.uniform(-4, 4, size=(n_wolves, n_features))

    def _binarise(pos: np.ndarray) -> np.ndarray:
        prob = _sigmoid(pos)
        return (rng.random(n_features) < prob).astype(int)

    def _binarise_det(pos: np.ndarray) -> np.ndarray:
        return (_sigmoid(pos) >= 0.5).astype(int)

    # Evaluate initial population
    fitness = np.array([
        _fitness(_binarise(positions[i]), X_tr, y_train, X_v, y_val,
                 epochs=fitness_epochs, batch_size=fitness_batch_size,
                 sample_frac=fitness_sample_frac, rng=rng, verbose=verbose)
        for i in range(n_wolves)
    ])

    sorted_idx = np.argsort(fitness)[::-1]
    alpha_pos = positions[sorted_idx[0]].copy()
    alpha_fit = fitness[sorted_idx[0]]
    beta_pos  = positions[sorted_idx[1]].copy()
    delta_pos = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

    best_local_mask = _binarise_det(alpha_pos)
    best_fit        = alpha_fit

    # Main loop
    for iteration in range(n_iter):
        a = 2.0 - 2.0 * (iteration / n_iter)

        if verbose:
            print(f"[BGWO] Iter {iteration + 1}/{n_iter} starting...")

        for i in range(n_wolves):
            new_pos = np.empty(n_features)
            for j in range(n_features):
                r1, r2 = rng.random(), rng.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                X1 = alpha_pos[j] - A1 * abs(C1 * alpha_pos[j] - positions[i, j])

                r1, r2 = rng.random(), rng.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                X2 = beta_pos[j] - A2 * abs(C2 * beta_pos[j] - positions[i, j])

                r1, r2 = rng.random(), rng.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                X3 = delta_pos[j] - A3 * abs(C3 * delta_pos[j] - positions[i, j])

                new_pos[j] = (X1 + X2 + X3) / 3.0
            positions[i] = new_pos

        fitness = np.array([
            _fitness(_binarise(positions[i]), X_tr, y_train, X_v, y_val,
                     epochs=fitness_epochs, batch_size=fitness_batch_size,
                     sample_frac=fitness_sample_frac, rng=rng, verbose=verbose)
            for i in range(n_wolves)
        ])

        sorted_idx = np.argsort(fitness)[::-1]
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitness[sorted_idx[0]]
        beta_pos  = positions[sorted_idx[1]].copy()
        delta_pos = positions[sorted_idx[2]].copy() if n_wolves > 2 else alpha_pos.copy()

        if alpha_fit > best_fit:
            best_fit        = alpha_fit
            best_local_mask = _binarise_det(alpha_pos)

        n_sel = int(best_local_mask.sum())
        if verbose:
            print(f"Iter {iteration + 1:3d}/{n_iter} | "
                  f"best_fitness={best_fit:.4f} | selected_features={n_sel}")

    # Map local mask back to full feature space
    if candidate_indices is not None:
        best_mask_full = np.zeros(n_features_full, dtype=int)
        best_mask_full[candidate_indices[best_local_mask == 1]] = 1
    else:
        best_mask_full = best_local_mask

    if verbose:
        print(f"\nBGWO complete. Selected {int(best_mask_full.sum())}/{n_features_full} features.")
    return best_mask_full


# ============================================================================
# 5.  LIME spot-checking
# ============================================================================

def lime_spot_check(
    model: Sequential,
    X_background: np.ndarray,
    X_instances: np.ndarray,
    feature_names: list[str],
    instance_indices: list[int] | None = None,
    n_samples: int = 1000,
    num_features: int = 10,
    verbose: bool = True,
) -> list[dict]:
    """
    Run LIME on a selection of instances and return their explanations.

    Particularly useful for inspecting misclassifications:
        wrong = np.where(y_pred != y_true)[0]
        lime_spot_check(model, X_train_bgwo, X_test_bgwo,
                        feature_names, instance_indices=wrong[:5].tolist())

    Parameters
    ----------
    model            : trained Keras model
    X_background     : representative background data (train set or subset)
    X_instances      : rows to explain (e.g. X_test_bgwo)
    feature_names    : list of column names (len == X_instances.shape[1])
    instance_indices : which rows of X_instances to explain; None → all
    n_samples        : LIME perturbation samples per instance
    num_features     : top features to include in each explanation
    verbose          : print a human-readable summary

    Returns
    -------
    explanations : list of dicts, one per instance:
        {
          'instance_index': int,
          'predicted_prob': float,
          'top_features'  : list of (feature_name, weight) tuples
        }
    """
    try:
        import lime  # noqa: F401
        import lime.lime_tabular  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lime is required for spot-checking.\n"
            "Install with:  pip install lime"
        ) from exc

    from lime.lime_tabular import LimeTabularExplainer

    def predict_fn(x: np.ndarray) -> np.ndarray:
        prob = model.predict(x, verbose=0).flatten()
        # LIME expects (n, n_classes) for classifiers
        return np.column_stack([1 - prob, prob])

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names,
        class_names=["normal", "attack"],
        mode="classification",
        random_state=42,
    )

    indices_to_explain = (
        instance_indices if instance_indices is not None
        else list(range(len(X_instances)))
    )

    results = []
    for idx in indices_to_explain:
        exp = explainer.explain_instance(
            X_instances[idx],
            predict_fn,
            num_features=num_features,
            num_samples=n_samples,
        )
        prob = predict_fn(X_instances[[idx]])[0, 1]
        top = exp.as_list()

        entry = {
            "instance_index": idx,
            "predicted_prob": float(prob),
            "top_features": top,
        }
        results.append(entry)

        if verbose:
            label = "attack" if prob >= 0.5 else "normal"
            print(f"\n[LIME] Instance {idx} → {label} (prob={prob:.3f})")
            for feat, weight in top:
                direction = "↑attack" if weight > 0 else "↓attack"
                print(f"  {feat:<45s} {direction}  {weight:+.4f}")

    return results
