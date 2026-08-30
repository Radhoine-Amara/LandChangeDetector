# LandChangeDetector

A QGIS plugin for detecting and mapping land use / land cover change between two satellite images of the same area, taken at different times.

Built as a university group project (Project 23), applied to Batna Province, Algeria using Sentinel-2 imagery — but the algorithms work on any two co-registered raster scenes.

## What it does

Give the plugin a "before" and "after" image of the same location and it will compute where the land has changed, using one of four detection methods:

| Method | What it measures | Good for |
|---|---|---|
| **Spectral Band Differencing** | Raw pixel-value change on a single band | Fast, simple change screening |
| **NDVI Differencing** | Change in vegetation index (`NDVI = (NIR - Red) / (NIR + Red)`) | Deforestation, drought, new crops, reforestation |
| **Change Vector Analysis (CVA)** | Magnitude + direction of change across all bands at once | Distinguishing *what kind* of change occurred (vegetation loss vs. urban growth vs. water) |
| **Random Forest (post-classification)** | Classifies land cover in each image separately, then compares the classes | Most detailed output; trained on ESA WorldCover labels, with spatial cross-validation and confidence filtering |

Optional cloud masking is supported via Sentinel-2 Scene Classification Layer (SCL) bands.

Results (change masks, statistics, classified rasters) are loaded straight into QGIS as new layers, and also written to disk as GeoTIFFs and CSV summaries.

## Installation

**Requirements:** QGIS ≥ 3.0, Python 3.11

1. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate landchange
   ```
2. Copy (or symlink) this folder into your QGIS plugins directory:
   ```
   ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/   # Linux
   %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\        # Windows
   ```
3. In QGIS: **Plugins → Manage and Install Plugins**, enable **Land Change Detector**.

## Usage

1. Open the plugin panel from the QGIS toolbar.
2. Select your **Before** and **After** images (Sentinel-2 B04/red band `.jp2`, or any matching raster pair).
3. *(Optional)* Enable cloud masking and select the corresponding SCL bands.
4. Choose a detection method.
5. Run the analysis — results are added to the QGIS layer panel and saved to an `output/` folder next to your input images.

## Project structure

```
land_change_detector.py           # Plugin entry point — toolbar/menu registration
land_change_detector_dialog.py    # UI dialog + background analysis worker
algorithms/                       # The four detection methods
  band_differencing.py
  ndvi_differencing.py
  cva.py
  random_forest.py
utils/                             # Raster I/O, tiling, stats, visualization helpers
test/                              # Unit tests (pytest / QGIS test harness)
help/                              # Sphinx documentation source
```

## Running tests

```bash
python -m pytest test/
```

## Author

- Mohammed Radhoine Amara

## License

No license file is currently included — all rights reserved by the authors unless stated otherwise.
