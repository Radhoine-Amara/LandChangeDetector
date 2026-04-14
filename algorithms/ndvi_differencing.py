"""
ndvi_differencing.py
--------------------
Method 2: NDVI Differencing

NDVI = (NIR - Red) / (NIR + Red)

More robust than band differencing because:
- Normalized ratio reduces seasonal/atmospheric effects
- Specifically targets vegetation changes
- Values always between -1 and +1 regardless of sensor

Math:
    NDVI_before = (NIR_b - Red_b) / (NIR_b + Red_b)
    NDVI_after  = (NIR_a - Red_a) / (NIR_a + Red_a)
    delta_NDVI  = NDVI_after - NDVI_before

Interpretation:
    delta_NDVI << 0  → vegetation LOST  (deforestation, drought, harvest)
    delta_NDVI >> 0  → vegetation GAINED (reforestation, new crops, irrigation)
    delta_NDVI ≈ 0   → no significant change
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_ndvi(red, nir):
    """
    Compute NDVI from Red and NIR bands.

    Parameters
    ----------
    red : np.ndarray  shape (rows, cols)  — Red band (B04)
    nir : np.ndarray  shape (rows, cols)  — NIR band (B08)

    Returns
    -------
    ndvi : np.ndarray
        Values between -1 and +1
        NaN where (NIR + Red) == 0 to avoid division by zero
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    denominator = nir + red

    # Avoid division by zero — set those pixels to NaN
    ndvi = np.where(
        denominator == 0,
        np.nan,
        (nir - red) / denominator
    )

    return ndvi.astype(np.float32)


def compute_ndvi_difference(red_before, nir_before, red_after, nir_after):
    """
    Compute NDVI for both dates and return the difference.

    Parameters
    ----------
    red_before, nir_before : np.ndarray — before image bands
    red_after,  nir_after  : np.ndarray — after image bands

    Returns
    -------
    ndvi_before  : np.ndarray  NDVI map at time 1
    ndvi_after   : np.ndarray  NDVI map at time 2
    delta_ndvi   : np.ndarray  signed difference (after - before)
    abs_delta    : np.ndarray  absolute difference
    """
    ndvi_before = compute_ndvi(red_before, nir_before)
    ndvi_after  = compute_ndvi(red_after,  nir_after)

    delta_ndvi = ndvi_after - ndvi_before
    abs_delta  = np.abs(delta_ndvi)

    return ndvi_before, ndvi_after, delta_ndvi, abs_delta


def otsu_threshold_ndvi(abs_delta):
    """
    Otsu threshold adapted for NDVI difference values (range 0 to 2).
    NaN pixels are excluded.
    """
    flat = abs_delta.flatten()
    flat = flat[~np.isnan(flat)]
    flat = flat[flat > 0]

    if len(flat) == 0:
        raise ValueError("No valid pixels for threshold calculation!")

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

        sum_bg  += hist[i] * bin_centers[i]
        mean_bg  = sum_bg / weight_bg
        mean_fg  = (total_sum - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh   = bin_centers[i]

    return best_thresh


def run_ndvi_differencing(red_before, nir_before, red_after, nir_after,
                           threshold=None, smooth=True):
    """
    Full NDVI differencing pipeline.

    Parameters
    ----------
    red_before, nir_before : np.ndarray — before bands (may contain NaN)
    red_after,  nir_after  : np.ndarray — after bands  (may contain NaN)
    threshold              : float or None (auto via Otsu if None)
    smooth                 : bool — apply Gaussian smoothing

    Returns
    -------
    results : dict with keys:
        'ndvi_before'     → NDVI map time 1
        'ndvi_after'      → NDVI map time 2
        'delta_ndvi'      → signed NDVI difference
        'abs_delta'       → absolute NDVI difference
        'change_mask'     → binary change map (True = changed)
        'gain_mask'       → vegetation GAINED (delta > +threshold)
        'loss_mask'       → vegetation LOST   (delta < -threshold)
        'threshold'       → threshold used
        'change_pct'      → % of valid pixels that changed
    """
    # Step 1: Compute NDVI and difference
    ndvi_b, ndvi_a, delta, abs_delta = compute_ndvi_difference(
        red_before, nir_before, red_after, nir_after
    )

    # Step 2: Track combined NaN mask
    nan_mask = np.isnan(abs_delta)

    # Step 3: Smooth (fill NaN temporarily)
    abs_delta_work = abs_delta.copy()
    abs_delta_work[nan_mask] = 0
    if smooth:
        abs_delta_work = gaussian_filter(abs_delta_work, sigma=1)
    abs_delta_work[nan_mask] = np.nan

    # Step 4: Threshold
    if threshold is None:
        threshold = otsu_threshold_ndvi(abs_delta_work)
        print(f"Auto threshold (Otsu): {threshold:.4f}")

    # Step 5: Build masks
    change_mask = np.where(nan_mask, False, abs_delta_work > threshold)
    gain_mask   = np.where(nan_mask, False, delta >  threshold)  # vegetation gained
    loss_mask   = np.where(nan_mask, False, delta < -threshold)  # vegetation lost

    # Step 6: Statistics
    valid_pixels = (~nan_mask).sum()
    change_pct   = (change_mask.sum() / valid_pixels) * 100 if valid_pixels > 0 else 0
    gain_pct     = (gain_mask.sum()   / valid_pixels) * 100 if valid_pixels > 0 else 0
    loss_pct     = (loss_mask.sum()   / valid_pixels) * 100 if valid_pixels > 0 else 0

    print(f"Valid pixels      : {valid_pixels:,}")
    print(f"Changed pixels    : {change_mask.sum():,}")
    print(f"Change %          : {change_pct:.2f}%")
    print(f"  → Vegetation gained : {gain_pct:.2f}%")
    print(f"  → Vegetation lost   : {loss_pct:.2f}%")

    # Step 7: NDVI stats
    valid_delta = delta[~nan_mask]
    print(f"\nNDVI before mean  : {ndvi_b[~np.isnan(ndvi_b)].mean():.3f}")
    print(f"NDVI after  mean  : {ndvi_a[~np.isnan(ndvi_a)].mean():.3f}")
    print(f"Delta NDVI  mean  : {valid_delta.mean():.3f}")

    return {
        'ndvi_before' : ndvi_b,
        'ndvi_after'  : ndvi_a,
        'delta_ndvi'  : delta,
        'abs_delta'   : abs_delta,
        'change_mask' : change_mask,
        'gain_mask'   : gain_mask,
        'loss_mask'   : loss_mask,
        'threshold'   : threshold,
        'change_pct'  : change_pct,
        'gain_pct'    : gain_pct,
        'loss_pct'    : loss_pct,
    }