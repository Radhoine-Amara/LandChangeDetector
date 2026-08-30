"""
utils/stats_utils.py
LandChangeDetector — Phase 5

Statistical analysis and export utilities.
Works with output dicts from all 4 change detection algorithms.

Author: Darius — 3rd Year AI Engineering
"""

import numpy as np
import pandas as pd
import os


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────

def _count_valid_pixels(mask_or_array):
    """Count non-NaN pixels in a 2D float array or True pixels in a bool mask."""
    if mask_or_array.dtype == bool:
        return int(np.sum(mask_or_array))
    return int(np.sum(~np.isnan(mask_or_array)))


# ─────────────────────────────────────────────────────────────
# 1. SINGLE-METHOD STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_change_statistics(results_dict, method_name):
    """
    Summarize change detection results into a pandas DataFrame.
    Works with output dicts from any of the 4 algorithms.

    Parameters
    ----------
    results_dict : dict
        Output from run_band_differencing(), run_ndvi_differencing(),
        run_cva(), or run_random_forest().
    method_name : str
        Human-readable name, e.g. 'Band Differencing', 'NDVI Differencing',
        'CVA', 'Random Forest'.

    Returns
    -------
    pd.DataFrame
        One row per metric. Columns: ['method', 'metric', 'value', 'unit'].
    """
    rows = []

    def add(metric, value, unit=""):
        rows.append({"method": method_name, "metric": metric,
                     "value": value, "unit": unit})

    # ── Universal metrics (all 4 algorithms expose these) ──────────────────
    change_mask = results_dict.get("change_mask")
    if change_mask is not None:
        total_pixels = change_mask.size
        changed_px   = int(np.sum(change_mask))
        unchanged_px = total_pixels - changed_px
        add("total_pixels",     total_pixels,                   "px")
        add("changed_pixels",   changed_px,                     "px")
        add("unchanged_pixels", unchanged_px,                   "px")

    change_pct = results_dict.get("change_pct")
    if change_pct is not None:
        add("change_pct", round(float(change_pct), 4), "%")

    threshold = results_dict.get("threshold")
    if threshold is not None:
        add("threshold", round(float(threshold), 6), "")

    # ── NDVI-specific ──────────────────────────────────────────────────────
    if "ndvi_before" in results_dict:
        ndvi_b = results_dict["ndvi_before"]
        ndvi_a = results_dict["ndvi_after"]
        delta  = results_dict["delta_ndvi"]

        add("ndvi_before_mean",  round(float(np.nanmean(ndvi_b)),  6), "")
        add("ndvi_after_mean",   round(float(np.nanmean(ndvi_a)),  6), "")
        add("delta_ndvi_mean",   round(float(np.nanmean(delta)),   6), "")
        add("delta_ndvi_std",    round(float(np.nanstd(delta)),    6), "")
        add("ndvi_before_std",   round(float(np.nanstd(ndvi_b)),   6), "")
        add("ndvi_after_std",    round(float(np.nanstd(ndvi_a)),   6), "")

    gain_pct = results_dict.get("gain_pct")
    loss_pct = results_dict.get("loss_pct")
    if gain_pct is not None:
        add("vegetation_gain_pct", round(float(gain_pct), 4), "%")
    if loss_pct is not None:
        add("vegetation_loss_pct", round(float(loss_pct), 4), "%")

    # ── CVA-specific ───────────────────────────────────────────────────────
    if "magnitude" in results_dict:
        mag = results_dict["magnitude"]
        add("magnitude_mean", round(float(np.nanmean(mag)), 4), "")
        add("magnitude_std",  round(float(np.nanstd(mag)),  4), "")
        add("magnitude_max",  round(float(np.nanmax(mag)),  4), "")

    type_stats = results_dict.get("type_stats")
    if type_stats is not None:
        for label, stats in type_stats.items():
            clean_label = label.lower().replace(" ", "_").replace("/", "_")
            add(f"cva_{clean_label}_px",
                int(stats.get("count", 0)), "px")
            add(f"cva_{clean_label}_pct",
                round(float(stats.get("pct", 0.0)), 4), "%")

    # ── Band Differencing-specific ─────────────────────────────────────────
    if "abs_diff" in results_dict and "magnitude" not in results_dict:
        abs_diff = results_dict["abs_diff"]
        add("abs_diff_mean", round(float(np.nanmean(abs_diff)), 4), "")
        add("abs_diff_std",  round(float(np.nanstd(abs_diff)),  4), "")
        add("abs_diff_max",  round(float(np.nanmax(abs_diff)),  4), "")

    # ── Random Forest-specific ─────────────────────────────────────────────
    if "accuracy" in results_dict:
        add("rf_validation_accuracy",
            round(float(results_dict["accuracy"]) * 100, 2), "%")

    if "class_names" in results_dict and "transition_matrix_pct" in results_dict:
        class_names = results_dict["class_names"]
        trans_pct   = results_dict["transition_matrix_pct"]
        # diagonal = stable (unchanged) pixels per class
        for i, name in enumerate(class_names):
            stable_pct = round(float(trans_pct[i, i]) * 100, 2)
            add(f"rf_stable_{name.lower().replace(' ','_')}_pct",
                stable_pct, "%")

    return pd.DataFrame(rows, columns=["method", "metric", "value", "unit"])


# ─────────────────────────────────────────────────────────────
# 2. AREA STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_area_statistics(change_mask, pixel_size_m=10):
    """
    Convert pixel counts to real-world areas (hectares and km²).

    Parameters
    ----------
    change_mask : np.ndarray (bool or float)
        Boolean change mask (True = changed).
        NaN pixels (cloud-masked) are excluded from all counts.
    pixel_size_m : int or float
        Pixel size in metres. Default 10 for Sentinel-2 R10m bands.

    Returns
    -------
    dict with keys:
        total_pixels, changed_pixels, unchanged_pixels, cloudy_pixels,
        pixel_area_m2, pixel_area_ha,
        changed_area_m2, changed_area_ha, changed_area_km2,
        unchanged_area_m2, unchanged_area_ha, unchanged_area_km2,
        change_pct  (of valid pixels only)
    """
    pixel_area_m2 = float(pixel_size_m ** 2)
    pixel_area_ha = pixel_area_m2 / 10_000.0

    total_pixels = int(change_mask.size)

    # Support both bool masks and float arrays with NaN
    if change_mask.dtype == bool:
        changed_px   = int(np.sum(change_mask))
        valid_px     = total_pixels
        cloudy_px    = 0
    else:
        valid_mask   = ~np.isnan(change_mask)
        valid_px     = int(np.sum(valid_mask))
        cloudy_px    = total_pixels - valid_px
        changed_px   = int(np.sum(change_mask[valid_mask].astype(bool)))

    unchanged_px = valid_px - changed_px
    change_pct   = (changed_px / valid_px * 100.0) if valid_px > 0 else 0.0

    def to_ha(px):
        return round(px * pixel_area_ha, 4)

    def to_km2(px):
        return round(px * pixel_area_m2 / 1_000_000.0, 6)

    return {
        "total_pixels":       total_pixels,
        "valid_pixels":       valid_px,
        "changed_pixels":     changed_px,
        "unchanged_pixels":   unchanged_px,
        "cloudy_pixels":      cloudy_px,
        "pixel_area_m2":      pixel_area_m2,
        "pixel_area_ha":      pixel_area_ha,
        "changed_area_m2":    round(changed_px   * pixel_area_m2, 2),
        "changed_area_ha":    to_ha(changed_px),
        "changed_area_km2":   to_km2(changed_px),
        "unchanged_area_m2":  round(unchanged_px * pixel_area_m2, 2),
        "unchanged_area_ha":  to_ha(unchanged_px),
        "unchanged_area_km2": to_km2(unchanged_px),
        "change_pct":         round(change_pct, 4),
    }


# ─────────────────────────────────────────────────────────────
# 3. EXPORT TO CSV
# ─────────────────────────────────────────────────────────────

def export_statistics_csv(stats_df, output_path):
    """
    Export change statistics DataFrame to a CSV file.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of compute_change_statistics() or compare_all_methods().
    output_path : str
        Full path to the output CSV, e.g.
        'output/change_statistics.csv'

    Returns
    -------
    str — absolute path to the saved file.

    Raises
    ------
    ValueError  if stats_df is None or empty.
    OSError     if the directory cannot be created.
    """
    if stats_df is None or stats_df.empty:
        raise ValueError("stats_df is empty — nothing to export.")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stats_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[stats_utils] Statistics exported → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
# 4. COMPARE ALL 4 METHODS
# ─────────────────────────────────────────────────────────────

def compare_all_methods(bd_results, ndvi_results, cva_results, rf_results):
    """
    Build a side-by-side comparison table of all 4 methods.

    Parameters
    ----------
    bd_results   : dict — output of run_band_differencing()
    ndvi_results : dict — output of run_ndvi_differencing()
    cva_results  : dict — output of run_cva()
    rf_results   : dict — output of run_random_forest()

    Returns
    -------
    pd.DataFrame
        Rows = metrics.  Columns = ['metric', 'unit',
        'Band Differencing', 'NDVI Differencing', 'CVA', 'Random Forest'].

    Usage
    -----
    df = compare_all_methods(bd_results, ndvi_results, cva_results, rf_results)
    print(df.to_string(index=False))
    export_statistics_csv(df, 'output/comparison.csv')
    """
    named = {
        "Band Differencing": bd_results,
        "NDVI Differencing": ndvi_results,
        "CVA":               cva_results,
        "Random Forest":     rf_results,
    }

    # ── Core comparable metrics for every method ──────────────────────────
    rows = []

    for method_name, res in named.items():
        if res is None:
            continue

        change_mask = res.get("change_mask")
        if change_mask is not None:
            total_px    = change_mask.size
            changed_px  = int(np.sum(change_mask))
            change_pct  = res.get("change_pct", changed_px / total_px * 100)
            threshold   = res.get("threshold", np.nan)

            rows.append({
                "method":         method_name,
                "change_pct":     round(float(change_pct),  4),
                "changed_pixels": changed_px,
                "total_pixels":   total_px,
                "threshold":      round(float(threshold), 6) if threshold is not None else np.nan,
            })

    comparison = pd.DataFrame(rows)

    # ── Add NDVI-specific columns ──────────────────────────────────────────
    if ndvi_results is not None:
        idx = comparison.index[comparison["method"] == "NDVI Differencing"]
        if len(idx):
            i = idx[0]
            comparison.loc[i, "ndvi_gain_pct"] = round(
                float(ndvi_results.get("gain_pct", np.nan)), 4)
            comparison.loc[i, "ndvi_loss_pct"] = round(
                float(ndvi_results.get("loss_pct", np.nan)), 4)
            comparison.loc[i, "delta_ndvi_mean"] = round(
                float(np.nanmean(ndvi_results["delta_ndvi"])), 6)

    # ── Add CVA change-type columns ────────────────────────────────────────
    if cva_results is not None:
        type_stats = cva_results.get("type_stats", {})
        idx = comparison.index[comparison["method"] == "CVA"]
        if len(idx):
            i = idx[0]
            for label, stats in type_stats.items():
                clean = "cva_" + label.lower().replace(" ", "_").replace("/", "_")
                comparison.loc[i, f"{clean}_pct"] = round(
                    float(stats.get("pct", 0.0)), 4)

    # ── Add Random Forest accuracy ─────────────────────────────────────────
    if rf_results is not None and "accuracy" in rf_results:
        idx = comparison.index[comparison["method"] == "Random Forest"]
        if len(idx):
            i = idx[0]
            comparison.loc[i, "rf_accuracy_pct"] = round(
                float(rf_results["accuracy"]) * 100, 2)

    if rf_results is not None and "transition_matrix_pct" in rf_results:
        idx = comparison.index[comparison["method"] == "Random Forest"]
        if len(idx):
            i = idx[0]
            trans_pct = rf_results["transition_matrix_pct"]
            class_names = rf_results.get("class_names", [])
            for j, name in enumerate(class_names):
                comparison.loc[i, f"rf_stable_{name.lower().replace(' ','_')}_pct"] = round(
                    float(trans_pct[j, j]), 2)

    return comparison


# ─────────────────────────────────────────────────────────────
# 5. AREA TABLE (ALL METHODS)
# ─────────────────────────────────────────────────────────────

def compute_all_areas(bd_results, ndvi_results, cva_results, rf_results,
                      pixel_size_m=10):
    """
    Compute real-world areas (ha, km²) for all 4 method change masks
    and return a summary DataFrame.

    Parameters
    ----------
    *_results    : result dicts from all 4 algorithms (pass None to skip).
    pixel_size_m : int — default 10 for Sentinel-2 R10m.

    Returns
    -------
    pd.DataFrame with columns:
        method, changed_area_ha, changed_area_km2,
        unchanged_area_ha, change_pct
    """
    named = {
        "Band Differencing": bd_results,
        "NDVI Differencing": ndvi_results,
        "CVA":               cva_results,
        "Random Forest":     rf_results,
    }
    rows = []
    for method_name, res in named.items():
        if res is None:
            continue
        mask = res.get("change_mask")
        if mask is None:
            continue
        areas = compute_area_statistics(mask, pixel_size_m=pixel_size_m)
        rows.append({
            "method":            method_name,
            "changed_area_ha":   areas["changed_area_ha"],
            "changed_area_km2":  areas["changed_area_km2"],
            "unchanged_area_ha": areas["unchanged_area_ha"],
            "change_pct":        areas["change_pct"],
            "valid_pixels":      areas["valid_pixels"],
        })
    return pd.DataFrame(rows)
