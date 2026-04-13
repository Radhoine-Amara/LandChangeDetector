"""
band_differencing.py
--------------------
Method 1: Spectral Band Differencing

The simplest change detection method.
Subtracts pixel values of the same band between two dates.
Large absolute differences indicate change.

Math:
    diff  = Band_After - Band_Before
    mask  = |diff| > threshold  → True means CHANGED
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_difference(band_before, band_after):
    """
    Subtract two single-band arrays.

    Parameters
    ----------
    band_before : np.ndarray  shape (rows, cols)  — pixel values at time 1
    band_after  : np.ndarray  shape (rows, cols)  — pixel values at time 2

    Returns
    -------
    diff : np.ndarray
        Signed difference (negative = decrease, positive = increase)
    abs_diff : np.ndarray
        Absolute difference (magnitude of change only)
    """

    if band_before.shape != band_after.shape:
        raise ValueError(
            f"Shape mismatch: before={band_before.shape}, after={band_after.shape}. "
            "Images must be the same size."
        )

    diff     = band_after.astype(np.float32) - band_before.astype(np.float32)
    abs_diff = np.abs(diff)

    return diff, abs_diff


def otsu_threshold(abs_diff):
    """
    Automatically find the best threshold using Otsu's method.

    Otsu's method finds the threshold that best separates
    two classes (changed vs unchanged) by minimizing
    within-class variance.

    Parameters
    ----------
    abs_diff : np.ndarray  — absolute difference image

    Returns
    -------
    threshold : float
    """

    # Flatten to 1D and remove zeros (no-change pixels dominate)
    flat = abs_diff.flatten()
    flat = flat[flat > 0]

    # Build histogram with 256 bins
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu's formula: maximize between-class variance
    total = hist.sum()
    best_thresh = 0
    best_variance = 0

    weight_bg = 0
    sum_bg = 0
    total_sum = np.sum(hist * bin_centers)

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]

        mean_bg = sum_bg / weight_bg
        mean_fg = (total_sum - sum_bg) / weight_fg

        # Between-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = bin_centers[i]

    return best_thresh


def apply_threshold(abs_diff, threshold=None, smooth=True):
    """
    Apply a threshold to produce a binary change mask.

    Parameters
    ----------
    abs_diff  : np.ndarray  — absolute difference image
    threshold : float or None
        If None → automatically determined using Otsu's method
    smooth    : bool
        If True → apply Gaussian smoothing before thresholding
        (reduces salt-and-pepper noise)

    Returns
    -------
    change_mask : np.ndarray (bool)
        True  → pixel changed
        False → pixel unchanged
    threshold : float
        The threshold value used (useful when auto-computed)
    """

    if smooth:
        abs_diff = gaussian_filter(abs_diff, sigma=1)

    if threshold is None:
        threshold = otsu_threshold(abs_diff)
        print(f"Auto threshold (Otsu): {threshold:.2f}")

    change_mask = abs_diff > threshold

    return change_mask, threshold


def run_band_differencing(band_before, band_after, threshold=None, smooth=True):
    """
    Full pipeline: difference → threshold → change mask.

    Parameters
    ----------
    band_before : np.ndarray  shape (rows, cols)
    band_after  : np.ndarray  shape (rows, cols)
    threshold   : float or None (auto if None)
    smooth      : bool

    Returns
    -------
    results : dict with keys:
        'diff'        → signed difference image
        'abs_diff'    → absolute difference image
        'change_mask' → binary change map (True = changed)
        'threshold'   → threshold value used
        'change_pct'  → percentage of pixels that changed
    """

    # Step 1: Compute difference
    diff, abs_diff = compute_difference(band_before, band_after)

    # Step 2: Apply threshold → binary mask
    change_mask, threshold = apply_threshold(abs_diff, threshold, smooth)

    # Step 3: Compute statistics
    change_pct = (change_mask.sum() / change_mask.size) * 100

    print(f"Changed pixels : {change_mask.sum():,}")
    print(f"Total pixels   : {change_mask.size:,}")
    print(f"Change %       : {change_pct:.2f}%")

    return {
        'diff'       : diff,
        'abs_diff'   : abs_diff,
        'change_mask': change_mask,
        'threshold'  : threshold,
        'change_pct' : change_pct
    }