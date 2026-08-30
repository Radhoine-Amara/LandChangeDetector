"""
utils/visualization.py
LandChangeDetector — Phase 5

Visualization utilities for change detection results.
Handles QGIS canvas rendering, matplotlib plots, and heatmaps.

Author: Darius — 3rd Year AI Engineering

NOTE: render_change_map_in_qgis() requires QGIS to be running.
All plot_*() functions work standalone with matplotlib/seaborn.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── QGIS imports — safe to import outside QGIS (just don't call them) ──────
try:
    from qgis.core import (
        QgsRasterLayer,
        QgsProject,
        QgsColorRampShader,
        QgsSingleBandPseudoColorRenderer,
        QgsRasterBandStats,
    )
    from PyQt5.QtWidgets import QApplication
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

# ── Internal raster save util ───────────────────────────────────────────────
try:
    from utils.raster_utils import save_raster
except ImportError:
    # Fallback: define inline so visualization.py works standalone
    from osgeo import gdal
    import numpy as np

    def save_raster(output_path, data, geo_transform, projection):
        driver = gdal.GetDriverByName("GTiff")
        if data.ndim == 2:
            rows, cols = data.shape
            n_bands = 1
        else:
            n_bands, rows, cols = data.shape
        dtype = gdal.GDT_Float32
        ds = driver.Create(output_path, cols, rows, n_bands, dtype)
        ds.SetGeoTransform(geo_transform)
        ds.SetProjection(projection)
        if n_bands == 1:
            band = ds.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            band.SetNoDataValue(np.nan)
        else:
            for i in range(n_bands):
                b = ds.GetRasterBand(i + 1)
                b.WriteArray(data[i].astype(np.float32))
        ds.FlushCache()
        ds = None


# ─────────────────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────────────────

# CVA change-type colours (matches cva.py class definitions)
CVA_COLORS = {
    0: "#888888",   # No change — Gray
    1: "#e63946",   # Vegetation loss — Red
    2: "#2dc653",   # Vegetation gain — Green
    3: "#f4a261",   # Urban/Soil gain — Orange
    4: "#457b9d",   # Water/Shadow — Blue
}
CVA_LABELS = {
    0: "No Change",
    1: "Vegetation Loss",
    2: "Vegetation Gain",
    3: "Urban/Soil Gain",
    4: "Water/Shadow",
}

# RF land-cover colours
RF_COLORS = {
    0: "#ffffff",   # Unclassified
    1: "#2dc653",   # Vegetation
    2: "#c1440e",   # Bare Soil / Urban
    3: "#457b9d",   # Water / Shadow
    4: "#f4d35e",   # Sparse Vegetation
}
RF_LABELS = {
    0: "Unclassified",
    1: "Vegetation",
    2: "Bare Soil/Urban",
    3: "Water/Shadow",
    4: "Sparse Vegetation",
}


# ─────────────────────────────────────────────────────────────
# 1. QGIS CANVAS RENDERING
# ─────────────────────────────────────────────────────────────

def render_change_map_in_qgis(change_mask, geo_transform, projection,
                               output_path, layer_name="Change Map"):
    """
    Save a change mask as GeoTIFF and add as a styled layer in QGIS canvas.

    Parameters
    ----------
    change_mask  : np.ndarray (bool or uint8)
        2D array where True / 1 = changed, 0 = unchanged.
        For CVA: pass change_type map (values 0–4).
    geo_transform : tuple (6-element)
        Spatial reference from load_band_crop().
    projection   : str
        WKT projection string from load_band_crop().
    output_path  : str
        Path where the GeoTIFF will be saved, e.g. 'output/change_map.tif'.
    layer_name   : str
        Name shown in QGIS Layers panel.

    Returns
    -------
    QgsRasterLayer  if QGIS is running, else None.

    Raises
    ------
    RuntimeError  if called outside a QGIS session.
    """
    if not QGIS_AVAILABLE:
        raise RuntimeError(
            "QGIS is not available. "
            "This function must be called from within QGIS (Plugin or Python console)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save as float32 GeoTIFF (0.0 = unchanged, 1.0 = changed)
    data_to_save = change_mask.astype(np.float32)
    save_raster(output_path, data_to_save, geo_transform, projection)

    # Load into QGIS
    layer = QgsRasterLayer(output_path, layer_name)
    if not layer.isValid():
        raise RuntimeError(f"QGIS could not load layer from {output_path}")

    # ── Apply binary colour ramp: 0 = transparent grey, 1 = red ───────────
    shader = QgsColorRampShader()
    shader.setColorRampType(QgsColorRampShader.Exact)

    color_items = [
        QgsColorRampShader.ColorRampItem(0, mcolors.to_qcolor("#cccccc"), "No Change"),
        QgsColorRampShader.ColorRampItem(1, mcolors.to_qcolor("#e63946"), "Changed"),
    ]
    shader.setColorRampItemList(color_items)

    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1)
    renderer.setClassificationMin(0)
    renderer.setClassificationMax(1)
    renderer.setShader(shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()

    QgsProject.instance().addMapLayer(layer)
    print(f"[visualization] Layer '{layer_name}' added to QGIS canvas.")
    return layer


def render_cva_map_in_qgis(change_type_map, geo_transform, projection,
                             output_path, layer_name="CVA Change Types"):
    """
    Save the CVA change-type map (0–4) as GeoTIFF and add a
    colour-classified layer to QGIS.

    Parameters
    ----------
    change_type_map : np.ndarray uint8
        Output 'change_type' array from run_cva().
    geo_transform, projection, output_path, layer_name — see render_change_map_in_qgis()
    """
    if not QGIS_AVAILABLE:
        raise RuntimeError("QGIS is not available.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_raster(output_path, change_type_map.astype(np.float32),
                geo_transform, projection)

    layer = QgsRasterLayer(output_path, layer_name)
    if not layer.isValid():
        raise RuntimeError(f"QGIS could not load layer from {output_path}")

    shader = QgsColorRampShader()
    shader.setColorRampType(QgsColorRampShader.Exact)

    color_items = []
    for class_val, hex_color in CVA_COLORS.items():
        label = CVA_LABELS[class_val]
        color_items.append(
            QgsColorRampShader.ColorRampItem(
                class_val, mcolors.to_qcolor(hex_color), label
            )
        )
    shader.setColorRampItemList(color_items)

    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1)
    renderer.setClassificationMin(0)
    renderer.setClassificationMax(4)
    renderer.setShader(shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()

    QgsProject.instance().addMapLayer(layer)
    print(f"[visualization] CVA layer '{layer_name}' added to QGIS canvas.")
    return layer


# ─────────────────────────────────────────────────────────────
# 2. TRANSITION MATRIX HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_transition_matrix(matrix_pct, class_names, output_path=None,
                            title="Land Cover Transition Matrix (%)"):
    """
    Render the Random Forest transition matrix as a seaborn heatmap
    with percentage annotations.

    Parameters
    ----------
    matrix_pct  : np.ndarray (n_classes × n_classes)
        Percentage values (0–100).  Row = class BEFORE, Col = class AFTER.
    class_names : list of str
        Labels for each class in the same order as matrix rows/cols.
    output_path : str or None
        If given, save figure to this path (e.g. 'output/transition_matrix.png').
    title       : str — Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.6),
                                    max(5, len(class_names) * 1.4)))

    # ── Mask the diagonal to highlight off-diagonal changes ─────────────
    mask_diag = np.eye(len(class_names), dtype=bool)

    sns.heatmap(
        matrix_pct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        mask=mask_diag,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of pixels (row class)", "shrink": 0.75},
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=np.max(matrix_pct[~mask_diag]) if np.any(~mask_diag) else 100,
    )

    # Overlay diagonal in green
    diag_data = np.full_like(matrix_pct, np.nan)
    np.fill_diagonal(diag_data, np.diag(matrix_pct))

    sns.heatmap(
        diag_data,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        mask=~mask_diag,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=100,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Class AFTER  (2025)", fontsize=11)
    ax.set_ylabel("Class BEFORE (2019)", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Transition matrix saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 3. METHOD COMPARISON (4-panel)
# ─────────────────────────────────────────────────────────────

def plot_method_comparison(bd_results, ndvi_results, cva_results, rf_results,
                            output_path=None):
    """
    Side-by-side comparison of all 4 change masks in one figure.

    Panels:
        Top-left  : Band Differencing change mask (binary)
        Top-right : NDVI Differencing — gain (green) / loss (red)
        Bottom-left: CVA change-type map (colour-coded by type)
        Bottom-right: Random Forest class-change map (binary)

    Parameters
    ----------
    *_results   : result dicts from each algorithm.  Pass None to show empty panel.
    output_path : str or None — save location.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "LandChangeDetector — Method Comparison\n"
        "Batna Province, Algeria  |  2019 → 2025",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── 1. Band Differencing ──────────────────────────────────────────────
    ax = axes[0, 0]
    if bd_results is not None:
        mask = bd_results["change_mask"].astype(float)
        mask[mask == 0] = np.nan                      # show only changed pixels
        ax.imshow(mask, cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        cp = bd_results.get("change_pct", 0)
        ax.set_title(f"Band Differencing\n{cp:.2f}% changed",
                     fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Band Differencing", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── 2. NDVI Differencing ───────────────────────────────────────────────
    ax = axes[0, 1]
    if ndvi_results is not None:
        gain = ndvi_results.get("gain_mask", np.zeros_like(ndvi_results["change_mask"]))
        loss = ndvi_results.get("loss_mask", np.zeros_like(ndvi_results["change_mask"]))
        # Build RGB: loss=red, gain=green, neither=transparent
        rows, cols = gain.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.float32)
        rgba[loss, 0] = 0.9   # red channel
        rgba[loss, 3] = 1.0   # alpha
        rgba[gain, 1] = 0.8   # green channel
        rgba[gain, 3] = 1.0
        ax.imshow(rgba, interpolation="nearest")
        gp = ndvi_results.get("gain_pct", 0)
        lp = ndvi_results.get("loss_pct", 0)
        ax.set_title(
            f"NDVI Differencing  ★ Most Reliable\n"
            f"Loss {lp:.2f}%  |  Gain {gp:.2f}%",
            fontsize=11, fontweight="bold"
        )
        gain_patch = mpatches.Patch(color="#00cc55", label="Veg. Gained")
        loss_patch = mpatches.Patch(color="#e63946", label="Veg. Lost")
        ax.legend(handles=[gain_patch, loss_patch], loc="lower right",
                  fontsize=8, framealpha=0.8)
    else:
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("NDVI Differencing", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── 3. CVA change-type ────────────────────────────────────────────────
    ax = axes[1, 0]
    if cva_results is not None:
        change_type = cva_results["change_type"].astype(float)
        # Build RGBA from CVA_COLORS lookup
        rows, cols = change_type.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.float32)
        for class_val, hex_col in CVA_COLORS.items():
            r, g, b = mcolors.to_rgb(hex_col)
            sel = (change_type == class_val)
            rgba[sel, 0] = r
            rgba[sel, 1] = g
            rgba[sel, 2] = b
            rgba[sel, 3] = 1.0 if class_val != 0 else 0.25
        ax.imshow(rgba, interpolation="nearest")
        cp = cva_results.get("change_pct", 0)
        ax.set_title(f"CVA — Change Vector Analysis\n{cp:.2f}% changed",
                     fontsize=11, fontweight="bold")
        legend_patches = [
            mpatches.Patch(color=CVA_COLORS[v], label=CVA_LABELS[v])
            for v in range(1, 5)
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=7, framealpha=0.8)
    else:
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("CVA", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── 4. Random Forest ──────────────────────────────────────────────────
    ax = axes[1, 1]
    if rf_results is not None:
        mask = rf_results["change_mask"].astype(float)
        mask[mask == 0] = np.nan
        ax.imshow(mask, cmap="Purples", interpolation="nearest", vmin=0, vmax=1)
        cp = rf_results.get("change_pct", 0)
        acc = rf_results.get("accuracy", None)
        acc_str = f"  |  Acc. {acc*100:.1f}%" if acc is not None else ""
        ax.set_title(f"Random Forest (Post-Classification){acc_str}\n{cp:.2f}% changed",
                     fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Random Forest", fontsize=11, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Method comparison saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 4. NDVI 3-PANEL VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_ndvi_comparison(ndvi_before, ndvi_after, delta_ndvi,
                          output_path=None,
                          title="NDVI Comparison — 2019 vs 2025"):
    """
    3-panel NDVI visualization: Before map | After map | Delta with histograms.

    Parameters
    ----------
    ndvi_before : np.ndarray (rows, cols) — NDVI at time T1 (2019)
    ndvi_after  : np.ndarray (rows, cols) — NDVI at time T2 (2025)
    delta_ndvi  : np.ndarray (rows, cols) — NDVI_after − NDVI_before
    output_path : str or None
    title       : str — Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.3)

    # ── Shared colour scale for before / after ────────────────────────────
    vmin_ndvi = -0.2
    vmax_ndvi =  0.8
    cmap_ndvi = "RdYlGn"

    # ── Top row: maps ──────────────────────────────────────────────────────
    ax_b  = fig.add_subplot(gs[0, 0])   # NDVI before
    ax_a  = fig.add_subplot(gs[0, 1])   # NDVI after
    ax_d  = fig.add_subplot(gs[0, 2])   # Delta NDVI

    # NDVI Before
    im_b = ax_b.imshow(ndvi_before, cmap=cmap_ndvi,
                        vmin=vmin_ndvi, vmax=vmax_ndvi,
                        interpolation="nearest")
    mean_b = np.nanmean(ndvi_before)
    ax_b.set_title(f"NDVI Before (2019)\nMean: {mean_b:.4f}",
                   fontsize=11, fontweight="bold")
    ax_b.axis("off")
    plt.colorbar(im_b, ax=ax_b, shrink=0.75, label="NDVI")

    # NDVI After
    im_a = ax_a.imshow(ndvi_after, cmap=cmap_ndvi,
                        vmin=vmin_ndvi, vmax=vmax_ndvi,
                        interpolation="nearest")
    mean_a = np.nanmean(ndvi_after)
    ax_a.set_title(f"NDVI After (2025)\nMean: {mean_a:.4f}",
                   fontsize=11, fontweight="bold")
    ax_a.axis("off")
    plt.colorbar(im_a, ax=ax_a, shrink=0.75, label="NDVI")

    # Delta NDVI — diverging palette centred at 0
    delta_abs_max = min(float(np.nanpercentile(np.abs(delta_ndvi), 99)), 0.6)
    im_d = ax_d.imshow(delta_ndvi, cmap="RdBu",
                        vmin=-delta_abs_max, vmax=delta_abs_max,
                        interpolation="nearest")
    mean_d = np.nanmean(delta_ndvi)
    ax_d.set_title(f"ΔNDVI (After − Before)\nMean: {mean_d:.4f}",
                   fontsize=11, fontweight="bold")
    ax_d.axis("off")
    cbar_d = plt.colorbar(im_d, ax=ax_d, shrink=0.75, label="ΔNDVI")
    cbar_d.set_ticks([-delta_abs_max, 0, delta_abs_max])
    cbar_d.set_ticklabels([f"−{delta_abs_max:.2f}", "0",
                            f"+{delta_abs_max:.2f}"])

    # ── Bottom row: histograms ──────────────────────────────────────────────
    ax_hb = fig.add_subplot(gs[1, 0])
    ax_ha = fig.add_subplot(gs[1, 1])
    ax_hd = fig.add_subplot(gs[1, 2])

    hist_kw = dict(bins=80, edgecolor="none", alpha=0.8)

    valid_b = ndvi_before[~np.isnan(ndvi_before)].ravel()
    valid_a = ndvi_after[~np.isnan(ndvi_after)].ravel()
    valid_d = delta_ndvi[~np.isnan(delta_ndvi)].ravel()

    ax_hb.hist(valid_b, color="#4CAF50", **hist_kw)
    ax_hb.axvline(mean_b, color="black", linewidth=1.5, linestyle="--")
    ax_hb.set_xlabel("NDVI", fontsize=9)
    ax_hb.set_ylabel("Pixel count", fontsize=9)
    ax_hb.set_xlim(vmin_ndvi, vmax_ndvi)

    ax_ha.hist(valid_a, color="#FF7043", **hist_kw)
    ax_ha.axvline(mean_a, color="black", linewidth=1.5, linestyle="--")
    ax_ha.set_xlabel("NDVI", fontsize=9)
    ax_ha.set_xlim(vmin_ndvi, vmax_ndvi)

    ax_hd.hist(valid_d, color="#5C6BC0", **hist_kw)
    ax_hd.axvline(mean_d, color="black", linewidth=1.5, linestyle="--")
    ax_hd.axvline(0, color="red", linewidth=1.0, linestyle=":")
    ax_hd.set_xlabel("ΔNDVI", fontsize=9)
    ax_hd.set_xlim(-delta_abs_max * 1.2, delta_abs_max * 1.2)

    # ── Annotation box ─────────────────────────────────────────────────────
    diff_str = f"NDVI change: {mean_b:.4f} → {mean_a:.4f}  (Δ = {mean_d:+.4f})"
    fig.text(0.5, -0.01, diff_str, ha="center", fontsize=10,
             color="#333333", style="italic")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] NDVI comparison saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 5. QUICK CHANGE SUMMARY BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_change_pct_bars(bd_results, ndvi_results, cva_results, rf_results,
                          output_path=None):
    """
    Horizontal bar chart comparing change % across all 4 methods.
    Useful as a quick visual summary for the notebook or report.

    Parameters
    ----------
    *_results   : result dicts (pass None to skip a method).
    output_path : str or None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    named = {
        "Band\nDifferencing": bd_results,
        "NDVI\nDifferencing": ndvi_results,
        "CVA": cva_results,
        "Random\nForest": rf_results,
    }
    colors_bar = ["#e63946", "#2dc653", "#f4a261", "#7b2d8b"]

    methods, values = [], []
    for name, res in named.items():
        if res is not None and "change_pct" in res:
            methods.append(name)
            values.append(round(float(res["change_pct"]), 2))

    if not methods:
        print("[visualization] No results to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(methods) * 1.1)))

    bars = ax.barh(methods, values,
                   color=colors_bar[:len(methods)],
                   height=0.55, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", ha="left", fontsize=10)

    ax.set_xlabel("Change Detected (%)", fontsize=11)
    ax.set_title("Change % by Detection Method\n"
                 "Batna Province, Algeria  |  2019 → 2025",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Bar chart saved → {output_path}")

    return fig
