"""
raster_utils.py
---------------
Utility functions for loading and handling raster data.
"""

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def load_raster(file_path):
    """
    Load a raster file and return its data as a NumPy array.

    Returns
    -------
    data         : np.ndarray  shape (bands, rows, cols)
    geo_transform: tuple       spatial position info
    projection   : str         coordinate reference system
    """
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

    if dataset is None:
        raise FileNotFoundError(f"Could not open raster: {file_path}")

    data = dataset.ReadAsArray().astype(np.float32)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    dataset = None

    return data, geo_transform, projection


def get_raster_info(file_path):
    """Print basic information about a raster file."""

    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

    if dataset is None:
        raise FileNotFoundError(f"Could not open raster: {file_path}")

    print("=" * 40)
    print(f"File     : {file_path.split('/')[-1]}")
    print(f"Size     : {dataset.RasterXSize} cols x {dataset.RasterYSize} rows")
    print(f"Bands    : {dataset.RasterCount}")
    print(f"CRS      : {dataset.GetProjection()[:60]}...")

    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        stats = band.ComputeStatistics(False)
        print(f"Band {i}   : min={stats[0]:.1f}, max={stats[1]:.1f}, mean={stats[2]:.1f}")

    print("=" * 40)
    dataset = None


def save_raster(output_path, data, geo_transform, projection):
    """Save a NumPy array as a GeoTIFF raster file."""

    if data.ndim == 2:
        bands, rows, cols = 1, data.shape[0], data.shape[1]
        data = data[np.newaxis, :]
    else:
        bands, rows, cols = data.shape

    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)

    for i in range(bands):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(data[i])
        out_band.FlushCache()

    out_dataset = None
    print(f"✅ Saved: {output_path}")


def normalize_band(band):
    """Normalize a band array to range [0, 1]."""

    band_min = np.nanmin(band)
    band_max = np.nanmax(band)

    if band_max - band_min == 0:
        return np.zeros_like(band, dtype=np.float32)

    return (band - band_min) / (band_max - band_min)
