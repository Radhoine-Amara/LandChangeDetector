"""
utils/tile_processor.py
LandChangeDetector — Tile-Based Full Image Processing

Divides a full Sentinel-2 image (10980×10980) into an N×N grid of tiles,
runs any change detection algorithm on each tile independently,
then stitches all results back into one seamless full-resolution output.

Architecture
────────────
Full image (10980×10980)
    │
    ▼  divide_into_tiles()
┌─────┬─────┬─────┐
│  0  │  1  │  2  │
├─────┼─────┼─────┤
│  3  │  4  │  5  │      ← each tile loaded independently → RAM safe
├─────┼─────┼─────┤
│  6  │  7  │  8  │
└─────┴─────┴─────┘
    │
    ▼  process_tile()  [one tile at a time, with overlap blending]
    │
    ▼  stitch_tiles()
Full result (10980×10980)  →  save_raster()

Key design decisions
────────────────────
- Overlap: each tile loads `overlap` extra pixels on all 4 sides.
  After processing, the overlap border is cropped before stitching.
  This prevents edge artifacts (e.g., Gaussian filter edge effects).
- Memory: only ONE tile is in RAM at a time — safe for 4 GB RAM laptops.
- Progress callback: optional function(done, total) for UI progress bars.
- All algorithms supported: pass any function(bands_before, bands_after) → dict.

Author: Darius — 3rd Year AI Engineering
"""

import os
import numpy as np
from osgeo import gdal

# ── Internal utils ──────────────────────────────────────────────────────────
try:
    from utils.raster_utils import load_band_crop, load_cloud_mask, save_raster
except ImportError:
    from raster_utils import load_band_crop, load_cloud_mask, save_raster


# ═══════════════════════════════════════════════════════════════════════════
# TILE GRID CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_tile_grid(image_width, image_height, n_tiles_x, n_tiles_y,
                       overlap=50):
    """
    Compute the (x_off, y_off, x_size, y_size) for every tile in an
    N×M grid, with optional overlap on each side.

    Parameters
    ----------
    image_width  : int — total image columns (e.g. 10980)
    image_height : int — total image rows    (e.g. 10980)
    n_tiles_x    : int — number of columns in the tile grid (e.g. 3)
    n_tiles_y    : int — number of rows    in the tile grid (e.g. 3)
    overlap      : int — extra pixels to load on each edge (default 50)
                   Prevents edge artifacts in smoothing / thresholding.

    Returns
    -------
    list of dicts, one per tile, with keys:
        tile_id       : int  (row * n_tiles_x + col, 0-indexed)
        row, col      : int  position in the grid
        x_off, y_off  : int  top-left pixel offset in the FULL image
        x_size, y_size: int  tile size INCLUDING overlap pixels
        x_off_inner   : int  x offset of the inner (no-overlap) region
        y_off_inner   : int  y offset of the inner (no-overlap) region
        x_size_inner  : int  width  of the inner region
        y_size_inner  : int  height of the inner region
        dest_x, dest_y: int  where this tile's inner result lands in
                             the stitched output array
    """
    base_w = image_width  // n_tiles_x
    base_h = image_height // n_tiles_y

    tiles = []
    tile_id = 0

    for row in range(n_tiles_y):
        for col in range(n_tiles_x):

            # ── Inner (core) region ────────────────────────────────────────
            inner_x = col * base_w
            inner_y = row * base_h

            # Last tile gets the remainder to cover the full image exactly
            inner_w = base_w if col < n_tiles_x - 1 else image_width  - inner_x
            inner_h = base_h if row < n_tiles_y - 1 else image_height - inner_y

            # ── Outer region (with overlap, clamped to image bounds) ───────
            outer_x = max(0, inner_x - overlap)
            outer_y = max(0, inner_y - overlap)
            outer_x2 = min(image_width,  inner_x + inner_w + overlap)
            outer_y2 = min(image_height, inner_y + inner_h + overlap)

            outer_w = outer_x2 - outer_x
            outer_h = outer_y2 - outer_y

            # ── Where does the inner region sit INSIDE the outer array? ────
            inner_in_outer_x = inner_x - outer_x
            inner_in_outer_y = inner_y - outer_y

            tiles.append({
                "tile_id":      tile_id,
                "row":          row,
                "col":          col,
                # What to pass to load_band_crop():
                "x_off":        outer_x,
                "y_off":        outer_y,
                "x_size":       outer_w,
                "y_size":       outer_h,
                # Crop after processing to remove overlap:
                "x_off_inner":  inner_in_outer_x,
                "y_off_inner":  inner_in_outer_y,
                "x_size_inner": inner_w,
                "y_size_inner": inner_h,
                # Where to paste into the output array:
                "dest_x":       inner_x,
                "dest_y":       inner_y,
            })
            tile_id += 1

    return tiles


def compute_scl_tile(tile, scale_factor=2):
    """
    Derive the SCL tile spec from a 10m tile spec.
    SCL is at 20m, so all pixel coordinates are divided by scale_factor=2.

    Parameters
    ----------
    tile         : dict — output of compute_tile_grid()
    scale_factor : int  — 2 for Sentinel-2 (10m → 20m)

    Returns
    -------
    dict with x_off, y_off, x_size, y_size for the SCL crop.
    """
    return {
        "x_off":  tile["x_off"]  // scale_factor,
        "y_off":  tile["y_off"]  // scale_factor,
        "x_size": max(1, tile["x_size"] // scale_factor),
        "y_size": max(1, tile["y_size"] // scale_factor),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE TILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_single_tile(tile, band_paths_before, band_paths_after,
                         scl_path_before, scl_path_after,
                         algorithm_fn, algorithm_kwargs=None,
                         enable_cloud_mask=True):
    """
    Load one tile from disk, apply cloud mask, run the algorithm,
    and return the inner (overlap-stripped) result.

    Parameters
    ----------
    tile              : dict — from compute_tile_grid()
    band_paths_before : list of str — [B02, B03, B04, B08] before paths
    band_paths_after  : list of str — same for after image
    scl_path_before   : str or None — SCL_20m.jp2 before path
    scl_path_after    : str or None — SCL_20m.jp2 after path
    algorithm_fn      : callable(bands_before, bands_after, **kwargs) → dict
                        Must be one of: run_band_differencing,
                        run_ndvi_differencing, run_cva, run_random_forest
    algorithm_kwargs  : dict or None — extra keyword args for the algorithm
    enable_cloud_mask : bool — whether to apply SCL masking

    Returns
    -------
    dict with:
        'change_mask'  : np.ndarray (inner_h, inner_w) bool
        'change_type'  : np.ndarray (inner_h, inner_w) uint8  [CVA only]
        'results_full' : the full results dict from the algorithm
        'geo_transform': adjusted geo_transform for this tile's inner region
        'projection'   : WKT string
    """
    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    crop = {
        "x_off":  tile["x_off"],
        "y_off":  tile["y_off"],
        "x_size": tile["x_size"],
        "y_size": tile["y_size"],
    }

    # ── Load all before bands ──────────────────────────────────────────────
    bands_before = []
    gt = proj = None
    for i, path in enumerate(band_paths_before):
        data, g, p = load_band_crop(path, **crop)
        bands_before.append(data.astype(np.float32))
        if i == 0:
            gt, proj = g, p

    if gt is None or proj is None:
        raise RuntimeError(
            "Failed to read geospatial metadata (GeoTransform/Projection) "
            "from the reference tile."
        )

    # ── Load all after bands ───────────────────────────────────────────────
    bands_after = []
    for path in band_paths_after:
        data, _, _ = load_band_crop(path, **crop)
        bands_after.append(data.astype(np.float32))

    # ── Cloud masking ──────────────────────────────────────────────────────
    if enable_cloud_mask and scl_path_before:
        scl_crop = compute_scl_tile(crop)
        target_shape = bands_before[0].shape

        _, valid_b = load_cloud_mask(scl_path_before, **scl_crop,
                                      target_size=target_shape)
        if scl_path_after and os.path.isfile(scl_path_after):
            _, valid_a = load_cloud_mask(scl_path_after, **scl_crop,
                                          target_size=target_shape)
            valid_both = valid_b & valid_a
        else:
            valid_both = valid_b   # after image assumed cloud-free

        def apply_mask(arr):
            return np.where(valid_both, arr, np.nan).astype(np.float32)

        bands_before = [apply_mask(b) for b in bands_before]
        bands_after  = [apply_mask(b) for b in bands_after]

    # ── Run algorithm ──────────────────────────────────────────────────────
    results = algorithm_fn(bands_before, bands_after, **algorithm_kwargs)

    # ── Crop overlap border from all output arrays ─────────────────────────
    xi  = tile["x_off_inner"]
    yi  = tile["y_off_inner"]
    xsz = tile["x_size_inner"]
    ysz = tile["y_size_inner"]

    def crop_inner(arr):
        if arr is None or not isinstance(arr, np.ndarray):
            return arr
        if arr.ndim == 2:
            return arr[yi: yi + ysz, xi: xi + xsz]
        if arr.ndim == 3:
            return arr[:, yi: yi + ysz, xi: xi + xsz]
        return arr

    cropped_results = {}
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            cropped_results[key] = crop_inner(val)
        else:
            cropped_results[key] = val   # scalars / dicts pass through

    # ── Adjust geo_transform to point to inner region ─────────────────────
    # gt = (top_left_x, px_width, 0, top_left_y, 0, px_height)
    adjusted_gt = (
        gt[0] + (tile["dest_x"]) * gt[1],   # shift x by inner dest
        gt[1],
        gt[2],
        gt[3] + (tile["dest_y"]) * gt[5],   # shift y by inner dest
        gt[4],
        gt[5],
    )

    return {
        "results_full":  cropped_results,
        "change_mask":   cropped_results.get("change_mask"),
        "change_type":   cropped_results.get("change_type"),
        "geo_transform": adjusted_gt,
        "projection":    proj,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STITCHING
# ═══════════════════════════════════════════════════════════════════════════

def stitch_tiles(tile_results, tile_grid, image_height, image_width,
                  output_key="change_mask", fill_value=0, dtype=np.uint8):
    """
    Stitch per-tile result arrays back into a single full-resolution array.

    Parameters
    ----------
    tile_results  : list of dicts — output of process_single_tile(), one per tile
    tile_grid     : list of dicts — output of compute_tile_grid()
    image_height  : int — full image rows    (e.g. 10980)
    image_width   : int — full image columns (e.g. 10980)
    output_key    : str — which key from results to stitch (default: 'change_mask')
    fill_value    : scalar — value for pixels not covered by any tile (rare)
    dtype         : numpy dtype — dtype of the output array

    Returns
    -------
    np.ndarray (image_height, image_width) — full stitched result
    """
    output = np.full((image_height, image_width), fill_value, dtype=dtype)

    for tile, tile_out in zip(tile_grid, tile_results):
        arr = tile_out["results_full"].get(output_key)
        if arr is None:
            continue

        dy = tile["dest_y"]
        dx = tile["dest_x"]
        h  = tile["y_size_inner"]
        w  = tile["x_size_inner"]

        # Guard against shape mismatch at image edges
        actual_h, actual_w = arr.shape[:2]
        h = min(h, actual_h)
        w = min(w, actual_w)

        output[dy: dy + h, dx: dx + w] = arr[:h, :w].astype(dtype)

    return output


# ═══════════════════════════════════════════════════════════════════════════
# MAIN HIGH-LEVEL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_tiled_analysis(band_paths_before,
                        band_paths_after,
                        scl_path_before,
                        scl_path_after,
                        algorithm_fn,
                        algorithm_kwargs=None,
                        output_dir="output",
                        output_prefix="result",
                        n_tiles_x=3,
                        n_tiles_y=3,
                        overlap=50,
                        enable_cloud_mask=True,
                        save_per_tile=False,
                        progress_callback=None):
    """
    Full pipeline: divide → process each tile → stitch → save.

    Parameters
    ----------
    band_paths_before : list[str]  [B02, B03, B04, B08] full paths
    band_paths_after  : list[str]  [B02, B03, B04, B08] full paths
    scl_path_before   : str or None  SCL_20m.jp2 before path
    scl_path_after    : str or None  SCL_20m.jp2 after path
    algorithm_fn      : callable  one of the 4 run_*() functions
    algorithm_kwargs  : dict or None  extra kwargs for the algorithm
    output_dir        : str  directory for all outputs
    output_prefix     : str  filename prefix (e.g. 'ndvi_differencing')
    n_tiles_x         : int  columns in tile grid (default 3 → 9 tiles)
    n_tiles_y         : int  rows    in tile grid (default 3 → 9 tiles)
    overlap           : int  overlap pixels per edge (default 50)
    enable_cloud_mask : bool
    save_per_tile     : bool  save individual tile GeoTIFFs (debug use)
    progress_callback : callable(done: int, total: int) or None
                        Called after each tile finishes.

    Returns
    -------
    dict with:
        'change_mask_full'  : np.ndarray (H, W) bool — stitched change mask
        'change_type_full'  : np.ndarray (H, W) uint8 — CVA types (or None)
        'geotiff_path'      : str  — path to saved full GeoTIFF
        'tile_results'      : list[dict]  — per-tile result dicts
        'tile_stats'        : list[dict]  — per-tile statistics
        'geo_transform'     : tuple(6) — geo_transform of full image
        'projection'        : str  — WKT projection

    Example
    -------
    from algorithms.ndvi_differencing import run_ndvi_differencing
    from utils.tile_processor import run_tiled_analysis

    results = run_tiled_analysis(
        band_paths_before=[".../B02.jp2", ".../B03.jp2",
                           ".../B04.jp2", ".../B08.jp2"],
        band_paths_after= [".../B02.jp2", ".../B03.jp2",
                           ".../B04.jp2", ".../B08.jp2"],
        scl_path_before=".../SCL_20m.jp2",
        scl_path_after=None,
        algorithm_fn=run_ndvi_differencing,
        algorithm_kwargs={"threshold": None, "smooth": True},
        output_dir="output/full_run",
        output_prefix="ndvi",
        n_tiles_x=3, n_tiles_y=3,
        progress_callback=lambda done, total: print(f"{done}/{total}")
    )
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Get full image dimensions from the first band ──────────────────────
    ds = gdal.Open(band_paths_before[0])
    if ds is None:
        raise FileNotFoundError(
            f"Could not open: {band_paths_before[0]}"
        )
    image_width  = ds.RasterXSize
    image_height = ds.RasterYSize
    full_gt      = ds.GetGeoTransform()
    full_proj    = ds.GetProjection()
    ds = None

    print(f"[tile_processor] Image size: {image_width} × {image_height}")
    print(f"[tile_processor] Tile grid:  {n_tiles_x} × {n_tiles_y} "
          f"= {n_tiles_x * n_tiles_y} tiles  (overlap={overlap}px)")

    # ── Build tile grid ────────────────────────────────────────────────────
    tile_grid = compute_tile_grid(
        image_width, image_height,
        n_tiles_x, n_tiles_y, overlap
    )
    total_tiles  = len(tile_grid)
    tile_results = []
    tile_stats   = []

    # ── Process each tile ──────────────────────────────────────────────────
    for i, tile in enumerate(tile_grid):
        print(f"[tile_processor] Processing tile {i+1}/{total_tiles}  "
              f"(row={tile['row']}, col={tile['col']})  "
              f"offset=({tile['x_off']},{tile['y_off']})  "
              f"size={tile['x_size']}×{tile['y_size']}")

        tile_out = process_single_tile(
            tile=tile,
            band_paths_before=band_paths_before,
            band_paths_after=band_paths_after,
            scl_path_before=scl_path_before,
            scl_path_after=scl_path_after,
            algorithm_fn=algorithm_fn,
            algorithm_kwargs=algorithm_kwargs,
            enable_cloud_mask=enable_cloud_mask,
        )
        tile_results.append(tile_out)

        # Per-tile statistics
        cm = tile_out.get("change_mask")
        if cm is not None:
            valid_px   = int(np.sum(~np.isnan(cm.astype(float))))
            changed_px = int(np.sum(cm))
            change_pct = (changed_px / valid_px * 100) if valid_px > 0 else 0.0
        else:
            valid_px = changed_px = 0
            change_pct = 0.0

        tile_stats.append({
            "tile_id":     tile["tile_id"],
            "row":         tile["row"],
            "col":         tile["col"],
            "valid_pixels":   valid_px,
            "changed_pixels": changed_px,
            "change_pct":     round(change_pct, 4),
        })
        print(f"  → change: {change_pct:.2f}%  "
              f"({changed_px} / {valid_px} valid px)")

        # Optional: save individual tile GeoTIFF
        if save_per_tile and cm is not None:
            tile_path = os.path.join(
                output_dir,
                f"{output_prefix}_tile_{tile['row']}_{tile['col']}.tif"
            )
            save_raster(
                tile_path,
                cm.astype(np.float32),
                tile_out["geo_transform"],
                tile_out["projection"]
            )

        if progress_callback:
            progress_callback(i + 1, total_tiles)

    # ── Stitch change_mask ─────────────────────────────────────────────────
    print("[tile_processor] Stitching change masks...")
    change_mask_full = stitch_tiles(
        tile_results, tile_grid,
        image_height, image_width,
        output_key="change_mask",
        fill_value=0,
        dtype=np.uint8,
    )

    # ── Stitch change_type (CVA only) ──────────────────────────────────────
    has_change_type = any(
        t["results_full"].get("change_type") is not None
        for t in tile_results
    )
    change_type_full = None
    if has_change_type:
        change_type_full = stitch_tiles(
            tile_results, tile_grid,
            image_height, image_width,
            output_key="change_type",
            fill_value=0,
            dtype=np.uint8,
        )

    # ── Save full stitched GeoTIFF ─────────────────────────────────────────
    geotiff_path = os.path.join(output_dir, f"{output_prefix}_full.tif")
    save_raster(
        geotiff_path,
        change_mask_full.astype(np.float32),
        full_gt, full_proj
    )
    print(f"[tile_processor] Saved full result → {geotiff_path}")

    if change_type_full is not None:
        cva_path = os.path.join(output_dir, f"{output_prefix}_full_types.tif")
        save_raster(
            cva_path,
            change_type_full.astype(np.float32),
            full_gt, full_proj
        )
        print(f"[tile_processor] Saved CVA types  → {cva_path}")

    # ── Summary stats ──────────────────────────────────────────────────────
    total_valid   = sum(s["valid_pixels"]   for s in tile_stats)
    total_changed = sum(s["changed_pixels"] for s in tile_stats)
    overall_pct   = (total_changed / total_valid * 100) if total_valid > 0 else 0.0

    print("\n" + "─" * 55)
    print(f"  TILE PROCESSING COMPLETE")
    print(f"  Tiles processed : {total_tiles}")
    print(f"  Total valid px  : {total_valid:,}")
    print(f"  Total changed   : {total_changed:,}  ({overall_pct:.2f}%)")
    print(f"  Output saved    : {geotiff_path}")
    print("─" * 55 + "\n")

    return {
        "change_mask_full":  change_mask_full.astype(bool),
        "change_type_full":  change_type_full,
        "geotiff_path":      geotiff_path,
        "tile_results":      tile_results,
        "tile_stats":        tile_stats,
        "geo_transform":     full_gt,
        "projection":        full_proj,
        "overall_change_pct": round(overall_pct, 4),
        "total_tiles":       total_tiles,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TILE STATISTICS REPORT
# ═══════════════════════════════════════════════════════════════════════════

def tile_stats_to_dataframe(tile_stats):
    """
    Convert tile_stats list into a pandas DataFrame for display or export.

    Parameters
    ----------
    tile_stats : list of dicts — from run_tiled_analysis()['tile_stats']

    Returns
    -------
    pd.DataFrame with columns:
        tile_id, row, col, valid_pixels, changed_pixels,
        change_pct, changed_area_ha, changed_area_km2
    """
    import pandas as pd

    PIXEL_AREA_HA  = (10 * 10) / 10_000     # 10m × 10m pixels
    PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

    rows = []
    for s in tile_stats:
        rows.append({
            "tile_id":         s["tile_id"],
            "row":             s["row"],
            "col":             s["col"],
            "valid_pixels":    s["valid_pixels"],
            "changed_pixels":  s["changed_pixels"],
            "change_pct":      s["change_pct"],
            "changed_area_ha": round(s["changed_pixels"] * PIXEL_AREA_HA,  2),
            "changed_area_km2":round(s["changed_pixels"] * PIXEL_AREA_KM2, 4),
        })

    df = pd.DataFrame(rows)
    return df


def print_tile_grid_summary(tile_stats, n_tiles_x, n_tiles_y):
    """
    Print a visual grid showing change % per tile — useful in notebooks.

    Example output:
    ┌──────────┬──────────┬──────────┐
    │ 12.34%   │  8.21%   │ 15.67%  │
    ├──────────┼──────────┼──────────┤
    │  9.88%   │ 11.02%   │ 13.44%  │
    ├──────────┼──────────┼──────────┤
    │  7.56%   │ 10.11%   │ 12.99%  │
    └──────────┴──────────┴──────────┘
    """
    grid = {}
    for s in tile_stats:
        grid[(s["row"], s["col"])] = s["change_pct"]

    cell_w = 10
    h_line_top    = "┌" + ("─" * cell_w + "┬") * (n_tiles_x - 1) + "─" * cell_w + "┐"
    h_line_mid    = "├" + ("─" * cell_w + "┼") * (n_tiles_x - 1) + "─" * cell_w + "┤"
    h_line_bottom = "└" + ("─" * cell_w + "┴") * (n_tiles_x - 1) + "─" * cell_w + "┘"

    print(h_line_top)
    for row in range(n_tiles_y):
        row_cells = []
        for col in range(n_tiles_x):
            pct   = grid.get((row, col), 0.0)
            label = f"{pct:.2f}%".center(cell_w)
            row_cells.append(label)
        print("│" + "│".join(row_cells) + "│")
        if row < n_tiles_y - 1:
            print(h_line_mid)
    print(h_line_bottom)
