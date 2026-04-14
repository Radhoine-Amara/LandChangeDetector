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


def process_in_tiles(file_path_before, file_path_after,
                     process_fn, tile_size=2000, overlap=50):
    """
    Process two full rasters tile by tile to save memory.
    See docstring above for full documentation.
    """
    ds_before = gdal.Open(file_path_before, gdal.GA_ReadOnly)
    ds_after  = gdal.Open(file_path_after,  gdal.GA_ReadOnly)

    cols = ds_before.RasterXSize
    rows = ds_before.RasterYSize
    result = np.zeros((rows, cols), dtype=np.float32)

    n_tiles_x = int(np.ceil(cols / tile_size))
    n_tiles_y = int(np.ceil(rows / tile_size))
    total_tiles = n_tiles_x * n_tiles_y

    print(f"Image size  : {cols} x {rows}")
    print(f"Tile size   : {tile_size} x {tile_size}")
    print(f"Total tiles : {total_tiles}")

    tile_count = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            x_start = tx * tile_size
            y_start = ty * tile_size
            x_read  = max(0, x_start - overlap)
            y_read  = max(0, y_start - overlap)
            x_end   = min(cols, x_start + tile_size + overlap)
            y_end   = min(rows, y_start + tile_size + overlap)
            width   = x_end - x_read
            height  = y_end - y_read

            tile_before = ds_before.GetRasterBand(1)\
                                   .ReadAsArray(x_read, y_read, width, height)\
                                   .astype(np.float32)
            tile_after  = ds_after.GetRasterBand(1)\
                                  .ReadAsArray(x_read, y_read, width, height)\
                                  .astype(np.float32)

            tile_result = process_fn(tile_before, tile_after)

            x_offset_in_tile = x_start - x_read
            y_offset_in_tile = y_start - y_read
            x_write_end = x_offset_in_tile + min(tile_size, cols - x_start)
            y_write_end = y_offset_in_tile + min(tile_size, rows - y_start)

            result[y_start:y_start + (y_write_end - y_offset_in_tile),
                   x_start:x_start + (x_write_end - x_offset_in_tile)] = \
                tile_result[y_offset_in_tile:y_write_end,
                            x_offset_in_tile:x_write_end]

            tile_count += 1
            print(f"  Tile {tile_count}/{total_tiles} "
                  f"({tile_count/total_tiles*100:.0f}%)", end='\r')

    print(f"\n✅ Done! Processed {total_tiles} tiles.")
    ds_before = None
    ds_after  = None
    return result


def load_band_crop(file_path, x_off=0, y_off=0, x_size=1000, y_size=1000):
    """
    Load a spatial crop of a single-band raster.
    Much more memory efficient than loading the full image.

    Parameters
    ----------
    file_path       : str
    x_off, y_off    : int  top-left pixel offset
    x_size, y_size  : int  number of pixels to read

    Returns
    -------
    data            : np.ndarray  shape (y_size, x_size)
    geo_transform   : tuple       adjusted for crop offset
    projection      : str
    """
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {file_path}")

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(x_off, y_off, x_size, y_size).astype(np.float32)

    gt_full = ds.GetGeoTransform()
    proj    = ds.GetProjection()

    gt_crop = (
        gt_full[0] + x_off * gt_full[1],
        gt_full[1],
        gt_full[2],
        gt_full[3] + y_off * gt_full[5],
        gt_full[4],
        gt_full[5]
    )

    ds = None
    return data, gt_crop, proj


def load_cloud_mask(scl_path, x_off=0, y_off=0, x_size=1000, y_size=1000,
                    target_size=None):
    """
    Load the Scene Classification Layer (SCL) and create a cloud mask.
    SCL is at 20m resolution, so we resize it to match 10m bands.

    Parameters
    ----------
    scl_path    : str   path to SCL_20m.jp2
    x_off       : int   offset (in 20m pixels → divide 10m offset by 2)
    y_off       : int   offset (in 20m pixels → divide 10m offset by 2)
    x_size      : int   size in 20m pixels
    y_size      : int   size in 20m pixels
    target_size : tuple (rows, cols) of the 10m band to resize to

    Returns
    -------
    cloud_mask : np.ndarray bool  True = cloudy pixel (should be masked)
    valid_mask : np.ndarray bool  True = valid pixel  (safe to use)
    """
    from scipy.ndimage import zoom

    ds = gdal.Open(scl_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open SCL: {scl_path}")

    # SCL is 20m, so divide 10m offsets by 2
    scl = ds.GetRasterBand(1).ReadAsArray(
        x_off, y_off, x_size, y_size
    ).astype(np.uint8)
    ds = None

    # Cloud pixel values in SCL
    cloud_values = [3, 8, 9, 10]  # shadows, med cloud, high cloud, cirrus

    # Build cloud mask
    cloud_mask = np.isin(scl, cloud_values)

    # Resize to match 10m band if needed (20m → 10m = scale by 2)
    if target_size is not None:
        scale_y = target_size[0] / scl.shape[0]
        scale_x = target_size[1] / scl.shape[1]
        cloud_mask = np.asarray(zoom(cloud_mask.astype(np.float32),
                         (scale_y, scale_x), order=0), dtype=bool)

    valid_mask = ~cloud_mask

    cloud_pct = cloud_mask.mean() * 100
    print(f"Cloud coverage in crop: {cloud_pct:.1f}%")
    print(f"Valid pixels          : {valid_mask.sum():,} / {valid_mask.size:,}")

    return cloud_mask, valid_mask
