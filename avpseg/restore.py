"""Match nnU-Net probability axes to the exported prediction."""

from __future__ import annotations

import itertools

import numpy as np


def match_probability_to_prediction(probability: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Resolve ambiguous nnU-Net probability axis order using the exported mask."""
    target_shape = tuple(prediction.shape)
    candidates: list[np.ndarray] = []
    if probability.shape == target_shape:
        candidates.append(probability)
    for axes in itertools.permutations(range(3)):
        candidate = np.transpose(probability, axes)
        if candidate.shape == target_shape:
            candidates.append(candidate)

    if not candidates:
        raise ValueError(f"Cannot match probability shape {probability.shape} to prediction shape {target_shape}")

    pred_mask = prediction > 0
    best_candidate = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        prob_mask = candidate >= 0.5
        overlap = np.logical_and(prob_mask, pred_mask).sum()
        denom = prob_mask.sum() + pred_mask.sum()
        score = 1.0 if denom == 0 else float(2 * overlap / denom)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate
