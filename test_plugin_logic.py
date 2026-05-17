"""
test_plugin_logic.py
====================
Headless tests for LandChangeDetector plugin logic.
No QGIS installation required — uses only numpy, GDAL, and the
algorithm/utility modules from this project.

Run with:
    conda run -n landchange python test_plugin_logic.py
"""

import os
import sys
import tempfile
import traceback

import numpy as np

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from osgeo import gdal
gdal.UseExceptions()

from utils.raster_utils import load_band_crop, save_raster
from algorithms.ndvi_differencing import run_ndvi_differencing
from algorithms.band_differencing import run_band_differencing

PASS = 0
FAIL = 0
_tmpdir = tempfile.mkdtemp(prefix="lcd_test_")


def _make_geotiff(path, data, dtype=gdal.GDT_Float32):
    """Write a synthetic single-band GeoTIFF to *path*."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, dtype)
    ds.SetGeoTransform((500000, 10, 0, 3900000, 0, -10))
    ds.SetProjection(
        'PROJCS["WGS 84 / UTM zone 31N",GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
        'PARAMETER["central_meridian",3],PARAMETER["scale_factor",0.9996],'
        'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
        'UNIT["metre",1]]'
    )
    ds.GetRasterBand(1).WriteArray(data.astype(np.float32))
    ds.FlushCache()
    ds = None


def _run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1
    except Exception:
        print(f"  [FAIL] {name}: unexpected exception")
        traceback.print_exc()
        FAIL += 1


# =============================================================================
# TEST 1 — load_band_crop on a synthetic GeoTIFF
# =============================================================================
def test1_load_band_crop():
    full = np.random.randint(0, 3000, (200, 200), dtype=np.int16).astype(np.float32)
    tif = os.path.join(_tmpdir, "test1_full.tif")
    _make_geotiff(tif, full)

    data, gt, proj = load_band_crop(tif, x_off=50, y_off=50, x_size=100, y_size=100)

    assert data.shape == (100, 100), f"Expected (100,100), got {data.shape}"
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert proj != "", "Projection string should not be empty"


# =============================================================================
# TEST 2 — run_ndvi_differencing on synthetic data
# =============================================================================
def test2_ndvi_differencing():
    rng = np.random.default_rng(42)
    red_b = rng.uniform(500, 2000, (100, 100)).astype(np.float32)
    nir_b = rng.uniform(2000, 4000, (100, 100)).astype(np.float32)
    red_a = rng.uniform(500, 2000, (100, 100)).astype(np.float32)
    nir_a = rng.uniform(2000, 4000, (100, 100)).astype(np.float32)

    result = run_ndvi_differencing(red_b, nir_b, red_a, nir_a)

    required_keys = {"ndvi_before", "ndvi_after", "delta_ndvi",
                     "change_mask", "change_pct"}
    missing = required_keys - result.keys()
    assert not missing, f"Missing keys: {missing}"

    pct = result["change_pct"]
    assert 0 <= pct <= 100, f"change_pct out of range: {pct}"
    assert result["change_mask"].dtype == bool, \
        f"change_mask must be bool, got {result['change_mask'].dtype}"


# =============================================================================
# TEST 3 — run_band_differencing on synthetic data
# =============================================================================
def test3_band_differencing():
    rng = np.random.default_rng(7)
    before = rng.uniform(500, 2500, (100, 100)).astype(np.float32)
    after  = rng.uniform(500, 2500, (100, 100)).astype(np.float32)

    result = run_band_differencing(before, after)

    assert result["change_mask"].dtype == bool, \
        f"change_mask must be bool, got {result['change_mask'].dtype}"
    assert isinstance(result["change_pct"], (float, np.floating)), \
        f"change_pct must be float, got {type(result['change_pct'])}"


# =============================================================================
# TEST 4 — save_raster produces a valid GeoTIFF
# =============================================================================
def test4_save_raster():
    data = np.random.rand(100, 100).astype(np.float32)
    gt   = (500000, 10, 0, 3900000, 0, -10)
    proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257]]]'
    out  = os.path.join(_tmpdir, "test4_output.tif")

    save_raster(out, data, gt, proj)

    assert os.path.isfile(out), "GeoTIFF was not created"
    ds = gdal.Open(out)
    assert ds is not None, "GDAL cannot open saved GeoTIFF"
    assert ds.RasterXSize == 100, f"Expected width 100, got {ds.RasterXSize}"
    assert ds.RasterYSize == 100, f"Expected height 100, got {ds.RasterYSize}"
    ds = None


# =============================================================================
# TEST 5 — NaN handling: cloud-masked pixels must not inflate change_pct
# =============================================================================
def test5_nan_handling():
    rng = np.random.default_rng(99)
    red_b = rng.uniform(500, 2000, (100, 100)).astype(np.float32)
    nir_b = rng.uniform(2000, 4000, (100, 100)).astype(np.float32)
    red_a = rng.uniform(500, 2000, (100, 100)).astype(np.float32)
    nir_a = rng.uniform(2000, 4000, (100, 100)).astype(np.float32)

    # Mask 20% of pixels as NaN (simulating cloud cover)
    mask = rng.random((100, 100)) < 0.20
    for arr in (red_b, nir_b, red_a, nir_a):
        arr[mask] = np.nan

    result = run_ndvi_differencing(red_b, nir_b, red_a, nir_a)

    change_mask = result["change_mask"]

    # NaN input pixels must be False in the change mask
    assert not change_mask[mask].any(), \
        "NaN-masked pixels must be False in change_mask"

    # change_pct must be a finite number
    pct = result["change_pct"]
    assert np.isfinite(pct), f"change_pct is not finite: {pct}"
    assert 0 <= pct <= 100, f"change_pct out of range: {pct}"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LandChangeDetector — Plugin Logic Tests")
    print("=" * 60)

    _run("Test 1: load_band_crop on synthetic GeoTIFF", test1_load_band_crop)
    _run("Test 2: run_ndvi_differencing on synthetic data", test2_ndvi_differencing)
    _run("Test 3: run_band_differencing on synthetic data", test3_band_differencing)
    _run("Test 4: save_raster produces valid GeoTIFF", test4_save_raster)
    _run("Test 5: NaN handling — cloud masked pixels", test5_nan_handling)

    print("=" * 60)
    print(f"Results: {PASS}/5 tests passed")
    if FAIL > 0:
        print(f"  {FAIL} test(s) FAILED — fix before proceeding.")
        sys.exit(1)
    else:
        print("  All tests PASSED.")
