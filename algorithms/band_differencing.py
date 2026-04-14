"""
band_differencing.py
--------------------
Method 1: Spectral Band Differencing
Supports NaN masking for cloud-free analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_difference(band_before, band_after):
    """
    Subtract two single-band arrays.
    NaN pixels (clouds) are preserved as NaN in output.
    """
    if band_before.shape != band_after.shape:
        raise ValueError(
            f"Shape mismatch: before={band_before.shape}, "
            f"after={band_after.shape}"
        )

    diff     = band_after.astype(np.float32) - band_before.astype(np.float32)
    abs_diff = np.abs(diff)

    return diff, abs_diff


def otsu_threshold(abs_diff):
    """
    Automatically find the best threshold using Otsu's method.
    NaN pixels are excluded from the calculation.
    """
    # Flatten and remove NaN and zeros
    flat = abs_diff.flatten()
    flat = flat[~np.isnan(flat)]   # ← remove NaN
    flat = flat[flat > 0]          # ← remove no-change pixels

    if len(flat) == 0:
        raise ValueError("No valid pixels to compute threshold!")

    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total      = hist.sum()
    best_thresh   = 0
    best_variance = 0
    weight_bg  = 0
    sum_bg     = 0
    total_sum  = np.sum(hist * bin_centers)

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg   += hist[i] * bin_centers[i]
        mean_bg   = sum_bg / weight_bg
        mean_fg   = (total_sum - sum_bg) / weight_fg
        variance  = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh   = bin_centers[i]

    return best_thresh


def apply_threshold(abs_diff, threshold=None, smooth=True):
    """
    Apply a threshold to produce a binary change mask.
    NaN pixels are always marked as unchanged (False).
    """
    # Work on a copy to avoid modifying original
    abs_diff_work = abs_diff.copy()

    # Temporarily fill NaN with 0 for smoothing
    nan_mask = np.isnan(abs_diff_work)
    abs_diff_work[nan_mask] = 0

    if smooth:
        abs_diff_work = gaussian_filter(abs_diff_work, sigma=1)

    # Restore NaN after smoothing
    abs_diff_work[nan_mask] = np.nan

    if threshold is None:
        threshold = otsu_threshold(abs_diff_work)
        print(f"Auto threshold (Otsu): {threshold:.2f}")

    # Threshold — NaN pixels = False (unchanged)
    change_mask = np.where(nan_mask, False, abs_diff_work > threshold)

    return change_mask, threshold


def run_band_differencing(band_before, band_after, threshold=None, smooth=True):
    """
    Full pipeline: difference → threshold → change mask.
    Handles NaN (cloud-masked) pixels correctly.
    """
    diff, abs_diff = compute_difference(band_before, band_after)
    change_mask, threshold = apply_threshold(abs_diff, threshold, smooth)

    # Statistics on VALID pixels only (exclude NaN)
    valid_pixels = (~np.isnan(abs_diff)).sum()
    change_pct   = (change_mask.sum() / valid_pixels) * 100 if valid_pixels > 0 else 0

    print(f"Valid pixels   : {valid_pixels:,}")
    print(f"Changed pixels : {change_mask.sum():,}")
    print(f"Change %       : {change_pct:.2f}% (of valid pixels)")

    return {
        'diff'       : diff,
        'abs_diff'   : abs_diff,
        'change_mask': change_mask,
        'threshold'  : threshold,
        'change_pct' : change_pct
    }
