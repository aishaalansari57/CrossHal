"""
classifier.py – HalluShift classifier training (same-dataset and cross-lingual).

Three variants are trained and compared in every run:

  Variant 1 — no_imbalance:
      Standard BCEWithLogitsLoss, threshold fixed at 0.5.
      No imbalance handling at all. This is the baseline.

  Variant 2 — pos_weight:
      BCEWithLogitsLoss with pos_weight = n_negative / n_positive.
      Penalises minority class misclassification. Threshold fixed at 0.5.

  Variant 3 — pos_weight_tuned:
      Same as variant 2 (pos_weight loss) but threshold is tuned on a
      validation split to maximise F1 instead of using the default 0.5.
      With ~85% hallucinated examples the optimal threshold is almost
      always well below 0.5.

Same-language experiments (1, 2, 3):
    StandardScaler fit on train, applied to test.

Cross-lingual experiments (4, 5, 6):
    QuantileTransformer fit independently on each language — fixes the
    scale mismatch between English and Arabic feature distributions.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class HalluShiftNet(nn.Module):
    """Simple feed-forward classifier used in the original HalluShift pipeline."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != 'hallucination']


def _to_tensors(X: np.ndarray, y: np.ndarray):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def _train_nn(X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 50, lr: float = 1e-3,
              pos_weight=None) -> HalluShiftNet:
    """Train one HalluShiftNet variant.

    pos_weight=None  → no imbalance handling (variant 1)
    pos_weight=tensor → weighted loss (variants 2 and 3)
    """
    X_t, y_t = _to_tensors(X_train, y_train)
    model     = HalluShiftNet(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        x = X_t
        x = model.dropout(model.relu(model.fc1(x)))
        x = model.dropout(model.relu(model.fc2(x)))
        logits = model.fc3(x)
        loss   = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    return model


def _get_probs(model: HalluShiftNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()


def _tune_threshold(model: HalluShiftNet,
                    X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find the threshold that maximises F1 on a validation set.

    Searches 0.05–0.95 in steps of 0.01. Falls back to 0.5 if no
    threshold produces a positive F1.
    """
    probs      = _get_probs(model, X_val)
    labels     = y_val.astype(int)
    best_thresh = 0.5
    best_f1     = 0.0
    for thresh in np.arange(0.05, 0.96, 0.01):
        score = f1_score(labels, (probs >= thresh).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, float(thresh)
    return best_thresh


def _compute_metrics(labels, preds, probs) -> dict:
    return {
        'accuracy':  accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'f1':        f1_score(labels, preds, zero_division=0),
        'auc_roc':   roc_auc_score(labels, probs),
        'pr_auc':    average_precision_score(labels, probs),
    }


def _evaluate(model: HalluShiftNet, X: np.ndarray,
              y: np.ndarray, threshold: float) -> dict:
    probs = _get_probs(model, X)
    return _compute_metrics(y.astype(int), (probs >= threshold).astype(int), probs)


def _print_results(results: dict, split_name: str,
                   thresh_tuned: float):
    """Print all three variants side by side.

    results keys: 'no_imbalance', 'pos_weight', 'pos_weight_tuned'
    """
    col_labels = {
        'no_imbalance':      'No handling  @0.5',
        'pos_weight':        'pos_weight   @0.5',
        'pos_weight_tuned':  f'pos_weight  @{thresh_tuned:.2f}',
    }
    keys = list(col_labels.keys())

    print(f"\n{'='*80}")
    print(f"  {split_name} — Three Imbalance Strategies")
    print(f"{'='*80}")
    header = f"  {'Metric':<12}" + "".join(f"  {col_labels[k]:>20}" for k in keys)
    print(header)
    print(f"  {'-'*12}" + "".join(f"  {'-'*20}" for _ in keys))

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'pr_auc']:
        vals     = {k: results[k][metric] for k in keys}
        best_key = max(vals, key=vals.get)
        row = f"  {metric:<12}"
        for k in keys:
            val_str = f"{vals[k]:.4f}"
            if k == best_key:
                val_str += " ◀"
            row += f"  {val_str:>20}"
        print(row)

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Same-language training (Experiments 1, 2, 3)
# ---------------------------------------------------------------------------

def train_combined_model(
    data: pd.DataFrame,
    num_layers: int,
    test_size: float = 0.25,
    seed: int = 42,
    save_dir: str | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """Train all three variants on a single dataset with an internal split.

    StandardScaler fit on train, applied to val and test.
    Validation set (20% of train) used only for threshold tuning.

    Returns:
        dict: {'no_imbalance': model, 'pos_weight': model, 'pos_weight_tuned': model}
              Note: pos_weight and pos_weight_tuned share the same trained weights —
              they differ only in the decision threshold applied at evaluation.
    """
    feature_cols = _get_feature_cols(data)
    X = data[feature_cols].values
    y = data['hallucination'].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    # Carve validation set for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    n_pos      = int(y_train.sum())
    n_neg      = int(len(y_train) - n_pos)
    pos_weight = _compute_pos_weight(y_train)
    print(f"\n  Class balance — hallucinated: {n_pos}, "
          f"not hallucinated: {n_neg}, "
          f"pos_weight: {pos_weight.item():.2f}")

    # ── Train variant 1: no imbalance handling ────────────────────────────────
    print("\n  Training variant 1 — no imbalance handling...")
    model_no = _train_nn(X_train, y_train, epochs=epochs, lr=lr, pos_weight=None)

    # ── Train variant 2 & 3: pos_weight (shared weights) ─────────────────────
    print("  Training variant 2 & 3 — pos_weight...")
    model_pw = _train_nn(X_train, y_train, epochs=epochs, lr=lr, pos_weight=pos_weight)

    # ── Tune threshold for variant 3 ──────────────────────────────────────────
    thresh_tuned = _tune_threshold(model_pw, X_val, y_val)
    print(f"  Tuned threshold (variant 3): {thresh_tuned:.2f}")

    # ── Evaluate all three variants ───────────────────────────────────────────
    results = {
        'no_imbalance':     _evaluate(model_no, X_test, y_test, 0.5),
        'pos_weight':       _evaluate(model_pw, X_test, y_test, 0.5),
        'pos_weight_tuned': _evaluate(model_pw, X_test, y_test, thresh_tuned),
    }

    _print_results(results, split_name="Same-Dataset Test",
                   thresh_tuned=thresh_tuned)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_no.state_dict(), os.path.join(save_dir, 'nn_no_imbalance.pth'))
        torch.save(model_pw.state_dict(), os.path.join(save_dir, 'nn_pos_weight.pth'))
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        pd.DataFrame([{'variant': 'pos_weight_tuned', 'threshold': thresh_tuned}]
                     ).to_csv(os.path.join(save_dir, 'thresholds.csv'), index=False)
        rows = [{'variant': k, **v} for k, v in results.items()]
        pd.DataFrame(rows).to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
        print(f"  Models, scaler, thresholds, and metrics saved to {save_dir}")

    return {
        'no_imbalance':     model_no,
        'pos_weight':       model_pw,
        'pos_weight_tuned': model_pw,  # same weights, different threshold
    }


# ---------------------------------------------------------------------------
# Cross-lingual training (Experiments 4, 5, 6)
# ---------------------------------------------------------------------------

def train_crosslingual_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_layers: int,
    seed: int = 42,
    save_dir: str | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """Train all three variants on train_df and evaluate on external test_df.

    QuantileTransformer fit independently on train and test languages.
    Validation set (20% of training data, same language) used for threshold tuning.

    Returns:
        tuple[dict, dict]: (models_dict, results_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    feature_cols = _get_feature_cols(train_df)

    missing = set(feature_cols) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test DataFrame is missing columns: {missing}")

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df['hallucination'].values
    X_test       = test_df[feature_cols].values
    y_test       = test_df['hallucination'].values

    # Carve validation set from training data for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    # Separate scalers per language
    qt_train = QuantileTransformer(output_distribution='normal', random_state=seed)
    qt_test  = QuantileTransformer(output_distribution='normal', random_state=seed)
    X_train  = qt_train.fit_transform(X_train)
    X_val    = qt_train.transform(X_val)      # val = same language as train
    X_test   = qt_test.fit_transform(X_test)  # test = different language

    n_pos      = int(y_train.sum())
    n_neg      = int(len(y_train) - n_pos)
    pos_weight = _compute_pos_weight(y_train)
    print(f"\n  Class balance — hallucinated: {n_pos}, "
          f"not hallucinated: {n_neg}, "
          f"pos_weight: {pos_weight.item():.2f}")

    # ── Train variant 1: no imbalance handling ────────────────────────────────
    print("\n  Training variant 1 — no imbalance handling...")
    model_no = _train_nn(X_train, y_train, epochs=epochs, lr=lr, pos_weight=None)

    # ── Train variant 2 & 3: pos_weight ──────────────────────────────────────
    print("  Training variant 2 & 3 — pos_weight...")
    model_pw = _train_nn(X_train, y_train, epochs=epochs, lr=lr, pos_weight=pos_weight)

    # ── Tune threshold for variant 3 ──────────────────────────────────────────
    thresh_tuned = _tune_threshold(model_pw, X_val, y_val)
    print(f"  Tuned threshold (variant 3): {thresh_tuned:.2f}")

    # ── Evaluate all three variants ───────────────────────────────────────────
    results = {
        'no_imbalance':     _evaluate(model_no, X_test, y_test, 0.5),
        'pos_weight':       _evaluate(model_pw, X_test, y_test, 0.5),
        'pos_weight_tuned': _evaluate(model_pw, X_test, y_test, thresh_tuned),
    }

    _print_results(results, split_name="Cross-Lingual Test",
                   thresh_tuned=thresh_tuned)

    models = {
        'no_imbalance':     model_no,
        'pos_weight':       model_pw,
        'pos_weight_tuned': model_pw,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_no.state_dict(), os.path.join(save_dir, 'nn_no_imbalance.pth'))
        torch.save(model_pw.state_dict(), os.path.join(save_dir, 'nn_pos_weight.pth'))
        with open(os.path.join(save_dir, 'scaler_train.pkl'), 'wb') as f:
            pickle.dump(qt_train, f)
        with open(os.path.join(save_dir, 'scaler_test.pkl'), 'wb') as f:
            pickle.dump(qt_test, f)
        pd.DataFrame([{'variant': 'pos_weight_tuned', 'threshold': thresh_tuned}]
                     ).to_csv(os.path.join(save_dir, 'thresholds.csv'), index=False)
        rows = [{'variant': k, **v} for k, v in results.items()]
        pd.DataFrame(rows).to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
        print(f"  Models, scalers, thresholds, and metrics saved to {save_dir}")

    return models, results


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model(save_dir: str, input_size: int,
               variant: str = 'nn_pos_weight') -> HalluShiftNet:
    """Reload a saved HalluShiftNet.

    variant: 'nn_no_imbalance' or 'nn_pos_weight'
    """
    model = HalluShiftNet(input_size=input_size)
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{variant}.pth')))
    model.eval()
    return model


def load_scaler(save_dir: str, name: str = 'scaler.pkl'):
    """Reload a saved scaler.

    name options:
        'scaler.pkl'       — same-language experiments
        'scaler_train.pkl' — cross-lingual train scaler
        'scaler_test.pkl'  — cross-lingual test scaler
    """
    with open(os.path.join(save_dir, name), 'rb') as f:
        return pickle.load(f)


def load_thresholds(save_dir: str) -> dict:
    """Reload the tuned threshold for the pos_weight_tuned variant."""
    df = pd.read_csv(os.path.join(save_dir, 'thresholds.csv'))
    return dict(zip(df['variant'], df['threshold']))