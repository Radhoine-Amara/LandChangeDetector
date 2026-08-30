"""
cva.py
------
Method 3: Change Vector Analysis (CVA)

Unlike band differencing (1 band) or NDVI (vegetation only),
CVA looks at ALL bands simultaneously as a vector.

For each pixel, we have a vector of band values:
    Before: V_before = [B02, B03, B04, B08]
    After:  V_after  = [B02, B03, B04, B08]

The CHANGE VECTOR is:
    ΔV = V_after - V_before = [ΔB02, ΔB03, ΔB04, ΔB08]

From this vector we extract:

1. MAGNITUDE  = √(ΔB02² + ΔB03² + ΔB04² + ΔB08²)
   → HOW MUCH changed (large = big change)

2. DIRECTION  = the angle of ΔV in spectral space
   → WHAT TYPE of change (vegetation loss vs urban gain vs water, etc.)

This gives us RICHER information than any single-band method.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_change_vectors(bands_before, bands_after):
    """
    Compute the change vector for each pixel across all bands.

    Parameters
    ----------
    bands_before : list of np.ndarray  [B02, B03, B04, B08] at time 1
    bands_after  : list of np.ndarray  [B02, B03, B04, B08] at time 2
                   Each array shape: (rows, cols)
                   May contain NaN for cloud-masked pixels.

    Returns
    -------
    delta        : np.ndarray  shape (n_bands, rows, cols)  — change vectors
    magnitude    : np.ndarray  shape (rows, cols)           — change magnitude
    """

    if len(bands_before) != len(bands_after):
        raise ValueError("bands_before and bands_after must have same number of bands")

    n_bands = len(bands_before)
    rows, cols = bands_before[0].shape

    # Stack into (n_bands, rows, cols) arrays
    stack_before = np.stack([b.astype(np.float32) for b in bands_before], axis=0)
    stack_after  = np.stack([b.astype(np.float32) for b in bands_after],  axis=0)

    # Change vector: shape (n_bands, rows, cols)
    delta = stack_after - stack_before

    # Magnitude: √(sum of squared differences across bands)
    # np.nansum ignores NaN — pixel is valid if at least some bands are valid
    magnitude = np.sqrt(np.nansum(delta ** 2, axis=0))

    # If ALL bands are NaN for a pixel → magnitude should be NaN too
    all_nan_mask = np.all(np.isnan(delta), axis=0)
    magnitude[all_nan_mask] = np.nan

    return delta, magnitude


def compute_direction_2d(delta, band_x_idx=2, band_y_idx=3):
    """
    Compute the 2D direction angle of the change vector.

    We pick 2 bands to define a 2D spectral plane:
        band_x_idx=2 → B04 (Red)   — x-axis
        band_y_idx=3 → B08 (NIR)   — y-axis

    This Red-NIR plane is the most informative for land use:

        Quadrant analysis (angle in degrees):
        ┌─────────────────────────────────────────────────┐
        │  Angle 90-180°  │  ΔRed<0, ΔNIR>0              │
        │  → Vegetation GAIN (more NIR, less Red)         │
        ├─────────────────────────────────────────────────┤
        │  Angle 0-90°    │  ΔRed>0, ΔNIR>0              │
        │  → Soil moisture increase / irrigation          │
        ├─────────────────────────────────────────────────┤
        │  Angle 270-360° │  ΔRed<0, ΔNIR<0              │
        │  → Water bodies / shadows                       │
        ├─────────────────────────────────────────────────┤
        │  Angle 180-270° │  ΔRed>0, ΔNIR<0              │
        │  → Vegetation LOSS (less NIR, more Red)         │
        └─────────────────────────────────────────────────┘

    Parameters
    ----------
    delta       : np.ndarray  shape (n_bands, rows, cols)
    band_x_idx  : int  index of the x-axis band (default: B04=Red)
    band_y_idx  : int  index of the y-axis band (default: B08=NIR)

    Returns
    -------
    direction : np.ndarray  shape (rows, cols)
        Angle in degrees [0, 360)
    """

    dx = delta[band_x_idx]   # ΔRed
    dy = delta[band_y_idx]   # ΔNIR

    # np.arctan2 returns angle in radians [-π, +π]
    # Convert to degrees and shift to [0, 360)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 360

    return angle_deg.astype(np.float32)


def classify_change_direction(direction, magnitude, magnitude_threshold):
    """
    Classify each changed pixel by the TYPE of change based on direction.

    Only classifies pixels where magnitude > threshold (real change).

    Classes:
        0 = No change       (magnitude below threshold)
        1 = Vegetation loss  (angle 180-270°: ΔRed>0, ΔNIR<0)
        2 = Vegetation gain  (angle  90-180°: ΔRed<0, ΔNIR>0)
        3 = Urban/soil gain  (angle   0-90°:  ΔRed>0, ΔNIR>0)
        4 = Water/shadow     (angle 270-360°: ΔRed<0, ΔNIR<0)

    Parameters
    ----------
    direction           : np.ndarray  shape (rows, cols)  degrees [0,360)
    magnitude           : np.ndarray  shape (rows, cols)
    magnitude_threshold : float

    Returns
    -------
    change_type : np.ndarray  uint8  shape (rows, cols)
    """

    change_type = np.zeros(direction.shape, dtype=np.uint8)

    # Only classify pixels that actually changed
    changed = (~np.isnan(magnitude)) & (magnitude > magnitude_threshold)

    # Assign class by direction quadrant
    change_type[changed & (direction >= 180) & (direction <  270)] = 1  # veg loss
    change_type[changed & (direction >=  90) & (direction <  180)] = 2  # veg gain
    change_type[changed & (direction >=   0) & (direction <   90)] = 3  # urban/soil
    change_type[changed & (direction >= 270) & (direction <  360)] = 4  # water/shadow

    return change_type


def otsu_threshold_magnitude(magnitude):
    """Otsu threshold on magnitude values, ignoring NaN and zeros."""

    flat = magnitude.flatten()
    flat = flat[~np.isnan(flat)]
    flat = flat[flat > 0]

    if len(flat) == 0:
        raise ValueError("No valid pixels for threshold!")

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


def run_cva(bands_before, bands_after,
            band_names=None, threshold=None, smooth=True):
    """
    Full CVA pipeline.

    Parameters
    ----------
    bands_before : list of np.ndarray  [B02, B03, B04, B08]
    bands_after  : list of np.ndarray  [B02, B03, B04, B08]
    band_names   : list of str         e.g. ['B02','B03','B04','B08']
    threshold    : float or None       auto via Otsu if None
    smooth       : bool

    Returns
    -------
    results : dict with keys:
        'delta'          → change vectors (n_bands, rows, cols)
        'magnitude'      → change magnitude (rows, cols)
        'direction'      → change direction in degrees (rows, cols)
        'change_mask'    → binary: True = changed
        'change_type'    → 0-4 classification map
        'threshold'      → magnitude threshold used
        'change_pct'     → % changed
        'type_stats'     → dict with % per change type
    """

    if band_names is None:
        band_names = [f'Band{i+1}' for i in range(len(bands_before))]

    print(f"Running CVA on {len(bands_before)} bands: {band_names}")

    # Step 1: Compute change vectors and magnitude
    delta, magnitude = compute_change_vectors(bands_before, bands_after)

    # Step 2: Smooth magnitude
    nan_mask = np.isnan(magnitude)
    mag_work = magnitude.copy()
    mag_work[nan_mask] = 0
    if smooth:
        mag_work = gaussian_filter(mag_work, sigma=1)
    mag_work[nan_mask] = np.nan

    # Step 3: Threshold
    if threshold is None:
        threshold = otsu_threshold_magnitude(mag_work)
        print(f"Auto threshold (Otsu): {threshold:.2f}")

    # Step 4: Binary change mask
    change_mask = np.where(nan_mask, False, mag_work > threshold)

    # Step 5: Direction and change type classification
    direction   = compute_direction_2d(delta, band_x_idx=2, band_y_idx=3)
    change_type = classify_change_direction(direction, mag_work, threshold)

    # Step 6: Statistics
    valid_pixels = (~nan_mask).sum()
    change_pct   = (change_mask.sum() / valid_pixels) * 100

    type_labels = {
        0: 'No change',
        1: 'Vegetation loss',
        2: 'Vegetation gain',
        3: 'Urban/Soil gain',
        4: 'Water/Shadow',
    }

    type_stats = {}
    for class_id, label in type_labels.items():
        if class_id == 0:
            continue
        count = (change_type == class_id).sum()
        pct   = (count / valid_pixels) * 100
        type_stats[label] = {'count': int(count), 'pct': float(pct)}
        print(f"  {label:20s}: {count:>8,} px  ({pct:.2f}%)")

    print(f"\nTotal change %  : {change_pct:.2f}%")
    print(f"Magnitude range : {np.nanmin(magnitude):.1f} – {np.nanmax(magnitude):.1f}")

    return {
        'delta'      : delta,
        'magnitude'  : magnitude,
        'direction'  : direction,
        'change_mask': change_mask,
        'change_type': change_type,
        'threshold'  : threshold,
        'change_pct' : change_pct,
        'type_stats' : type_stats,
    }