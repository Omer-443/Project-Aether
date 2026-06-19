"""
cde_formatter.py â€” Project Aether Data Pipeline
================================================
Converts raw, irregularly-spaced CMS claim event records into
Hermite cubic spline coefficients consumed by `torchcde`.

Key function: `prepare_cms_for_cde(df)`
  1. Parses claim timestamps â†’ continuous physical time (days since epoch).
  2. Normalises all clinical features.
  3. Prepends the time channel to the feature matrix (required by torchcde).
  4. Calls `torchcde.hermite_cubic_coefficients_with_backward_differences`
     to produce the spline coefficients that faithfully represent an
     irregularly-sampled path as a smooth, continuous function X(t).

Why Hermite cubic?
  - It has well-defined derivatives at every knot point.
  - It is causal: coefficients at t_i depend only on data up to t_i
    (critical for real-time inference).
  - `torchcde` natively supports it and it integrates well with rk4/dopri5.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchcde

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema expected in the CMS-like DataFrame
# ---------------------------------------------------------------------------

# Continuous clinical features used as CDE input channels (excluding time)
FEATURE_COLS: List[str] = [
    "allowed_amount",       # USD amount allowed by payer
    "billed_amount",        # USD amount billed by provider
    "drg_weight",           # CMS DRG relative weight (case complexity)
    "length_of_stay",       # Inpatient days
    "icd_chapter",          # Primary ICD-10 chapter (integer-encoded)
    "procedure_count",      # Number of distinct procedures in the claim
    "prior_denial_rate",    # Historical denial rate for this provider (0-1)
    "payer_score",          # Composite payer-strictness index (0-1)
]

TIME_COL: str = "claim_date"           # datetime column
EPOCH: pd.Timestamp = pd.Timestamp("2015-01-01")  # reference zero-point


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_cms_for_cde(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    time_col: str = TIME_COL,
    device: str = "cpu",
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Convert a CMS-like claim DataFrame into Hermite cubic spline coefficients.

    The function handles:
        - Non-uniform temporal gaps between claims (the whole point of CDEs).
        - Missing feature values (forward-fill then mean-fill as fallback).
        - Feature normalization to zero-mean / unit-variance.
        - Prepending the continuous time channel as the first feature so
          the CDE vector field receives X(t) = [t, feature_1, ..., feature_n].

    Args:
        df:           DataFrame with at least `time_col` and `feature_cols`.
        feature_cols: Which columns to use (defaults to FEATURE_COLS).
        time_col:     Name of the datetime column.
        device:       "cpu" or "cuda".
        normalize:    Whether to z-score normalise the feature channels.

    Returns:
        coeffs:    Hermite cubic coefficients tensor,
                   shape (1, n_intervals, 4 * n_channels).
                   Ready to be passed to `torchcde.CubicSpline(coeffs)`.
        times_t:   1-D tensor of physical time points (days), shape (n,).
        stats:     numpy array (n_features, 2) â€” [mean, std] per feature
                   (useful for de-normalising predictions later).

    Raises:
        ValueError: If df has fewer than 2 rows or required columns are absent.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    _validate_dataframe(df, time_col, feature_cols)

    # â”€â”€ 1. Sort chronologically â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # â”€â”€ 2. Convert timestamps â†’ continuous float (days since EPOCH) â”€â”€â”€â”€â”€â”€
    times_np: np.ndarray = (
        (df[time_col] - EPOCH).dt.total_seconds() / 86_400.0
    ).values.astype(np.float32)

    # Guard against duplicate timestamps (torchcde requires strictly monotone t)
    times_np = _make_strictly_monotone(times_np)

    # â”€â”€ 3. Extract & clean features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    features: np.ndarray = df[feature_cols].values.astype(np.float32)

    # Forward-fill NaN, then fill any remaining with column mean
    features = _forward_fill(features)
    col_means = np.nanmean(features, axis=0)
    nan_mask = np.isnan(features)
    features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # â”€â”€ 4. Normalise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    stats = np.stack([features.mean(axis=0), features.std(axis=0) + 1e-8], axis=1)
    if normalize:
        features = (features - stats[:, 0]) / stats[:, 1]

    # â”€â”€ 5. Prepend time channel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # X(t) = [t, f_1(t), ..., f_n(t)]  â€” shape (n, 1 + n_features)
    t_channel = times_np.reshape(-1, 1)
    X_np = np.concatenate([t_channel, features], axis=1)   # (n, n_channels)

    # â”€â”€ 6. Build Hermite cubic spline coefficients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    times_t = torch.tensor(times_np, dtype=torch.float32, device=device)

    # torchcde expects (batch, time, channels) â€” add batch dim
    X_batch = X_t.unsqueeze(0)                              # (1, n, n_channels)

    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
        x=X_batch,
        t=times_t,
    )
    # coeffs shape: (1, n-1, 4 * n_channels)

    logger.info(
        "[CDE Formatter] Prepared %d claim events -> coeffs shape %s | "
        "time span %.1f-%.1f days",
        len(df), tuple(coeffs.shape), float(times_t[0]), float(times_t[-1]),
    )

    return coeffs, times_t, stats


def coeffs_from_claim_events(
    events: List[dict],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Convenience wrapper: convert a list of JSON-style claim event dicts
    (as received from the FastAPI endpoint) into CDE coefficients.

    Each dict must contain keys matching FEATURE_COLS + TIME_COL.

    Args:
        events: List of claim event dicts from the API request body.
        device: Torch device string.

    Returns:
        Same as `prepare_cms_for_cde`.
    """
    df = pd.DataFrame(events)
    return prepare_cms_for_cde(df, device=device)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: List[str],
) -> None:
    if len(df) < 2:
        raise ValueError(
            f"DataFrame must have at least 2 rows for spline interpolation; "
            f"got {len(df)}."
        )
    missing = [c for c in [time_col] + feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _make_strictly_monotone(
    times: np.ndarray, epsilon: float = 1e-2
) -> np.ndarray:
    """
    Ensure strictly increasing time values by adding tiny increments
    to any duplicates.  This is required by the spline interpolator.
    """
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + epsilon
    return times


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Pandas-style forward-fill on a 2-D numpy array (row = time step)."""
    df = pd.DataFrame(arr)
    df.ffill(inplace=True)
    return df.values.copy()
