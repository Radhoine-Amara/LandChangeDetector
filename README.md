# LandChangeDetector 🛰️

A QGIS plugin for bi-temporal land use/land cover change detection using Sentinel-2 imagery.

## Methods
- Spectral Band Differencing
- NDVI Differencing  
- Change Vector Analysis (CVA)
- Random Forest Post-Classification

## Setup
```bash
conda env create -f environment.yml
conda activate landchange
pip install pb_tool
```

## Team
- Member 1 — UI + Integration
- Member 2 — Algorithms (Band diff, NDVI)
- Member 3 — Algorithms (CVA, Random Forest)

## Data
Place Sentinel-2 images in `data/before/` and `data/after/` (not tracked by git).
