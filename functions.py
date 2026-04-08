"""
functions.py – Core HalluShift feature extraction utilities.
"""

import itertools
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tensor / distribution helpers
# ---------------------------------------------------------------------------

def normalize_as_distribution(tensor):
    """Reshape tensor and apply softmax to obtain a probability distribution."""
    tensor = tensor.view(-1)
    return F.softmax(tensor, dim=-1)


def wasserstein_dist(p, q):
    """Wasserstein distance between two distribution tensors."""
    p = p.to(torch.float32).cpu().numpy()
    q = q.to(torch.float32).cpu().numpy()
    return wasserstein_distance(p, q)


def cosine_similarity(tensor1, tensor2):
    """Cosine similarity between two tensors."""
    tensor1 = tensor1.view(-1, tensor1.size(-1)).to(torch.float32)
    tensor2 = tensor2.view(-1, tensor2.size(-1)).to(torch.float32)
    return F.cosine_similarity(tensor1, tensor2).item()


# ---------------------------------------------------------------------------
# Internal-state feature extraction (hidden states / attentions)
# ---------------------------------------------------------------------------

def plot_internal_state_2(outputs, num_layers, state="hidden"):
    """Extract Wasserstein distances and cosine similarities from consecutive
    internal model states, averaged across all generated tokens.

    Args:
        outputs: HuggingFace model output with hidden_states / attentions.
        num_layers (int): Total number of transformer layers in the model.
        state (str): "hidden" or "attention".

    Returns:
        list[float]: [(num_layers//2 - 1) * 2] values —
                     first half are Wasserstein distances, second half cosines.
    """
    results = []
    # index_sums size is computed after we know the actual vec length,
    # so we initialise it lazily on the first valid result.
    index_sums = None

    if state == "hidden":
        for tup in outputs.hidden_states:
            # Clamp indices to the actual tuple length — some models (e.g. Qwen3)
            # return fewer hidden state tensors than num_layers would suggest.
            tup_len = len(tup)
            vec = [normalize_as_distribution(tup[i])
                   for i in range(2, num_layers + 1, 2) if i < tup_len]
            if len(vec) < 2:
                continue
            div = [wasserstein_dist(vec[i], vec[i + 1]) for i in range(len(vec) - 1)]
            div.extend(cosine_similarity(vec[i], vec[i + 1]) for i in range(len(vec) - 1))
            results.append(div)
    else:
        for tup in outputs.attentions:
            # Same bounds check for attention tuples.
            tup_len = len(tup)
            vec = [normalize_as_distribution(tup[i])
                   for i in range(1, num_layers, 2) if i < tup_len]
            if len(vec) < 2:
                continue
            div = [wasserstein_dist(vec[i], vec[i + 1]) for i in range(len(vec) - 1)]
            div.extend(cosine_similarity(vec[i], vec[i + 1]) for i in range(len(vec) - 1))
            results.append(div)

    if not results:
        # No valid tuples found — return zeros with the expected length
        return [0.0] * ((num_layers // 2) - 1) * 2

    # Initialise index_sums from the actual result length (may differ from
    # num_layers for models like Qwen3 that have fewer attention tensors).
    index_sums = [0.0] * len(results[0])
    for res, i in itertools.product(results, range(len(results[0]))):
        index_sums[i] += res[i]

    return [s / len(results) for s in index_sums]


# ---------------------------------------------------------------------------
# Token-probability features
# ---------------------------------------------------------------------------

def probability_function(output):
    """Return per-token max and min probabilities from model logits.

    Returns:
        [max_prob_list, min_prob_list]
    """
    max_probs, min_probs = [], []
    for logit in output.logits:
        probs = F.softmax(logit[0], dim=0)
        max_probs.append(probs.max().item())
        min_probs.append(probs.min().item())
    return [max_probs, min_probs]


def normalized_entropy(prob_list):
    """Normalised Shannon entropy of a probability list (0–1)."""
    entropy = -np.sum([p * np.log(p) for p in prob_list if p > 0])
    max_entropy = np.log(len(prob_list))
    return entropy / max_entropy if max_entropy > 0 else 0


def count_low_probs(prob_list, threshold=0.1):
    """Count elements below *threshold*."""
    return sum(p < threshold for p in prob_list)


def count_high_probs(prob_list, threshold=0.9):
    """Count elements above *threshold*."""
    return sum(p > threshold for p in prob_list)


def probability_gradients(prob_list):
    """Absolute first-order differences of consecutive probabilities."""
    return [abs(prob_list[i + 1] - prob_list[i]) for i in range(len(prob_list) - 1)]


def mean_gradient(prob_list):
    """Mean absolute gradient of probabilities."""
    g = probability_gradients(prob_list)
    return np.mean(g) if g else 0


def max_gradient(prob_list):
    """Max absolute gradient of probabilities."""
    g = probability_gradients(prob_list)
    return max(g) if g else 0


def percentile(prob_list, q):
    """q-th percentile of a probability list."""
    return np.percentile(prob_list, q)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def truncate_after_words(text, num_words=128):
    """Truncate *text* to at most *num_words* words."""
    return " ".join(text.split()[:num_words])


# ---------------------------------------------------------------------------
# Feature engineering / data preparation
# ---------------------------------------------------------------------------

def data_preparation(df_1, df_2, num_layers):
    """Engineer features and merge with hallucination labels.

    Modifies *df_1* in-place for the probability columns, then concatenates
    the hallucination column from *df_2*.

    The probability list columns are always the last two numeric columns
    before any named columns — we locate them by position from the actual
    DataFrame shape rather than computing from num_layers. This makes the
    function robust to models (e.g. Qwen3) whose attention tuple length
    differs from num_layers, which changes the total feature count.

    Args:
        df_1 (pd.DataFrame): Raw HalluShift feature DataFrame.
        df_2 (pd.DataFrame): DataFrame with a 'hallucination' column.
        num_layers (int): Number of model layers (kept for compatibility).

    Returns:
        pd.DataFrame: Feature-engineered DataFrame ready for classification.
    """
    # The raw feature DataFrame has integer column indices.
    # The probability lists are always the last two integer-indexed columns
    # (max_probs at index -2, min_probs at index -1).
    int_cols = [c for c in df_1.columns if isinstance(c, int)]
    prob_max_col = int_cols[-2]
    prob_min_col = int_cols[-1]

    temp_max = df_1[prob_max_col].copy()
    temp_min = df_1[prob_min_col].copy()

    # Maximum spread (Mps)
    df_1[prob_min_col] = df_1.apply(
        lambda row: max(a - b for a, b in zip(row[prob_max_col], row[prob_min_col])),
        axis=1,
    )
    df_1[prob_max_col] = df_1[prob_max_col].apply(min)

    # Entropy features
    df_1['norm_entropy_max'] = temp_max.apply(normalized_entropy)
    df_1['norm_entropy_min'] = temp_min.apply(normalized_entropy)

    # Low-prob count features
    df_1['low_prob_count_max'] = temp_max.apply(lambda x: count_low_probs(x, threshold=0.1))
    df_1['low_prob_count_min'] = temp_min.apply(lambda x: count_low_probs(x, threshold=0.1))

    # Gradient features
    df_1['mean_grad_max'] = temp_max.apply(mean_gradient)
    df_1['mean_grad_min'] = temp_min.apply(mean_gradient)

    # Percentile features
    df_1['p25_max'] = temp_max.apply(lambda x: percentile(x, 25))
    df_1['p50_max'] = temp_max.apply(lambda x: percentile(x, 50))
    df_1['p75_max'] = temp_max.apply(lambda x: percentile(x, 75))

    if df_1.shape[0] != df_2.shape[0]:
        raise ValueError("DataFrames have different lengths.")

    return pd.concat([df_1, df_2['hallucination']], axis=1)