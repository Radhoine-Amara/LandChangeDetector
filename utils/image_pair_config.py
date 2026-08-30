"""
utils/image_pair_config.py
==========================
Project 23 — LandChangeDetector  (Batna Province, Algeria)

Image pair configuration system.

Supports running the full pipeline with multiple image pairs
for the same-season validation experiment.

Usage
-----
    from utils.image_pair_config import ImagePairConfig, get_pair_config

    # Cross-seasonal pair (existing)
    cross = get_pair_config('cross_seasonal')

    # Same-season pair (after downloading June 2024)
    same = get_pair_config('same_season')

    # Use a config in the notebook:
    CROP = cross.crop
    bands_before, bands_after, valid_both = cross.load_bands_and_mask()
"""

import os
import platform
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List


def _find_data_root() -> Path:
    """Auto-discover data directory across machines and OS."""

    # Priority 1: Environment variable
    env_path = os.environ.get('LAND_CHANGE_DATA')
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Priority 2: Relative to this file's location (utils/ -> project root)
    here = Path(__file__).parent.parent
    candidate = here / 'data'
    if candidate.exists():
        return candidate

    # Priority 3: Relative to current working directory
    candidate = Path.cwd() / 'data'
    if candidate.exists():
        return candidate

    # Priority 4: Machine-specific hardcoded fallback
    if platform.system() == 'Windows':
        fallback = Path(r'D:\School Shit\Semester 6\Group Project\land_change_detector\data')
    else:
        fallback = Path('/home/darius/qgis_projects/land_change_detector/data')

    if fallback.exists():
        print(f"WARNING: Using hardcoded fallback path: {fallback}")
        print("    Set LAND_CHANGE_DATA env var to avoid this warning.")
        return fallback

    raise FileNotFoundError(
        "Could not find data directory. "
        "Set the LAND_CHANGE_DATA environment variable to your data folder path."
    )


DATA_ROOT = _find_data_root()


@dataclass
class ImagePairConfig:
    """
    Complete configuration for one Sentinel-2 image pair.

    All path fields accept environment variables:
        '~/...' is expanded automatically.
    """
    name         : str
    description  : str

    # ── Band paths — Before image ─────────────────────────────────────────
    B02_B        : str   # Blue  10m
    B03_B        : str   # Green 10m
    B04_B        : str   # Red   10m
    B08_B        : str   # NIR   10m
    SCL_B        : str   # Scene Classification Layer 20m

    # ── Band paths — After image ──────────────────────────────────────────
    B02_A        : str
    B03_A        : str
    B04_A        : str
    B08_A        : str
    SCL_A        : str

    # ── Crop window (shared by both dates) ────────────────────────────────
    crop         : Dict  = field(default_factory=lambda: {
                               'x_off': 4000, 'y_off': 4000,
                               'x_size': 2000, 'y_size': 2000})

    # ── WorldCover (optional) ─────────────────────────────────────────────
    worldcover_paths : Optional[List[str]] = None

    # ── Output ────────────────────────────────────────────────────────────
    output_dir   : str   = 'output/batna'

    # ── Metadata ─────────────────────────────────────────────────────────
    date_before  : str   = ''
    date_after   : str   = ''
    season_before: str   = ''
    season_after : str   = ''

    def __post_init__(self):
        for attr in ['B02_B','B03_B','B04_B','B08_B','SCL_B',
                     'B02_A','B03_A','B04_A','B08_A','SCL_A',
                     'output_dir']:
            setattr(self, attr, os.path.expanduser(getattr(self, attr)))
        if self.worldcover_paths:
            self.worldcover_paths = [os.path.expanduser(p)
                                      for p in self.worldcover_paths]

    @property
    def scl_crop(self):
        """SCL crop at 20m = half the 10m crop offsets and sizes."""
        c = self.crop
        return {
            'x_off'  : c['x_off']  // 2,
            'y_off'  : c['y_off']  // 2,
            'x_size' : c['x_size'] // 2,
            'y_size' : c['y_size'] // 2,
        }

    def check_files(self):
        """Verify all required files exist. Returns (ok, missing_list)."""
        files = [self.B02_B, self.B03_B, self.B04_B, self.B08_B, self.SCL_B,
                 self.B02_A, self.B03_A, self.B04_A, self.B08_A, self.SCL_A]
        if self.worldcover_paths:
            files += self.worldcover_paths
        missing = [f for f in files if not os.path.exists(f)]
        return len(missing) == 0, missing

    def load_bands_and_mask(self):
        """
        Load all bands for both dates and compute the combined valid mask.

        Returns
        -------
        bands_before : list[np.ndarray]  [B02,B03,B04,B08]  float32 with NaN
        bands_after  : list[np.ndarray]
        valid_both   : np.ndarray bool  (y_size, x_size)
        """
        from utils.raster_utils import load_band_crop, load_cloud_mask

        c    = self.crop
        sc   = self.scl_crop
        size = (c['y_size'], c['x_size'])

        print(f"Loading {self.name} ...")
        print(f"  Before: {self.date_before} ({self.season_before})")
        print(f"  After:  {self.date_after}  ({self.season_after})")

        def load(path):
            d, _, _ = load_band_crop(path, **c)
            return d.astype(np.float32)

        blue_b,  green_b,  red_b,  nir_b  = (load(self.B02_B), load(self.B03_B),
                                               load(self.B04_B), load(self.B08_B))
        blue_a,  green_a,  red_a,  nir_a  = (load(self.B02_A), load(self.B03_A),
                                               load(self.B04_A), load(self.B08_A))

        _, valid_b = load_cloud_mask(self.SCL_B, target_size=size, **sc)
        _, valid_a = load_cloud_mask(self.SCL_A, target_size=size, **sc)
        valid_both = valid_b & valid_a

        def apply_mask(band, mask):
            out = band.copy()
            out[~mask] = np.nan
            return out

        bands_before = [apply_mask(b, valid_both)
                        for b in [blue_b, green_b, red_b, nir_b]]
        bands_after  = [apply_mask(b, valid_both)
                        for b in [blue_a, green_a, red_a, nir_a]]

        n_valid = int(valid_both.sum())
        pct     = n_valid / valid_both.size * 100
        print(f"  Valid pixels: {n_valid:,} / {valid_both.size:,} ({pct:.1f}%)")
        print(f"  Cloud before: {(~valid_b).sum():,} px  "
              f"({(~valid_b).mean()*100:.1f}%)")
        print(f"  Cloud after:  {(~valid_a).sum():,} px  "
              f"({(~valid_a).mean()*100:.1f}%)")

        return bands_before, bands_after, valid_both

    def summary(self):
        """Print a human-readable configuration summary."""
        ok, missing = self.check_files()
        print(f"\n{'='*60}")
        print(f"ImagePairConfig: {self.name}")
        print(f"  {self.description}")
        print(f"  Before : {self.date_before}  ({self.season_before})")
        print(f"  After  : {self.date_after}   ({self.season_after})")
        print(f"  Crop   : x_off={self.crop['x_off']}, y_off={self.crop['y_off']}, "
              f"{self.crop['x_size']}x{self.crop['y_size']} px")
        print(f"  WorldCover: {self.worldcover_paths or 'None (KMeans fallback)'}")
        print(f"  Output : {self.output_dir}")
        status = "OK" if ok else f"MISSING {len(missing)} files"
        print(f"  Files  : {status}")
        if missing:
            for m in missing:
                print(f"    MISSING: {m}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# PRESET CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_pair_config(pair_name: str,
                    data_root: Optional[str] = None) -> ImagePairConfig:
    """
    Return a pre-built ImagePairConfig for the specified pair.

    Parameters
    ----------
    pair_name : str
        'cross_seasonal'  — June 2019 vs August 2025  (existing project pair)
        'same_season'     — June 2019 vs June/July 2024  (new validation pair)
    data_root : str or None
        Root directory where Sentinel-2 SAFE folders are stored.
        Defaults to DATA_ROOT auto-discovered above.

    Returns
    -------
    ImagePairConfig
    """
    root = Path(data_root) if data_root else DATA_ROOT

    def safe_band(safe_folder: str, date_str: str, band: str, res: str = 'R10m') -> str:
        """Build path to a specific band inside a .SAFE archive."""
        safe_path = root / safe_folder
        granule_dir = safe_path / 'GRANULE'
        if not granule_dir.exists():
            return str(safe_path / 'GRANULE' / '???' / 'IMG_DATA' / res / f'{band}.jp2')
        granules = [d for d in granule_dir.iterdir() if d.is_dir()]
        if not granules:
            return str(safe_path / 'GRANULE' / '???' / 'IMG_DATA' / res / f'{band}.jp2')
        img_data = granules[0] / 'IMG_DATA' / res
        if img_data.exists():
            matches = [f for f in img_data.iterdir()
                       if band in f.name and f.suffix == '.jp2']
            if matches:
                return str(matches[0])
        return str(img_data / f'T31SGV_{date_str}_{band}_10m.jp2')

    WC_PATHS = [
        str(root / 'worldcover' / 'ESA_WorldCover_10m_2021_v200_N33E006_Map.tif'),
        str(root / 'worldcover' / 'ESA_WorldCover_10m_2021_v200_N33E003_Map.tif'),
    ]

    # ── Cross-seasonal pair (existing project pair) ───────────────────────
    if pair_name == 'cross_seasonal':
        SAFE_B = 'S2B_MSIL2A_20190601T103629_N0500_R008_T31SGV_20230601T000000.SAFE'
        SAFE_A = 'S2B_MSIL2A_20250828T103559_N0511_R008_T31SGV_20250828T130000.SAFE'
        return ImagePairConfig(
            name        = 'Cross-Seasonal (Jun 2019 vs Aug 2025)',
            description = 'Original project pair — seasonal mismatch present',
            B02_B = safe_band(SAFE_B, '20190601T103629', 'B02'),
            B03_B = safe_band(SAFE_B, '20190601T103629', 'B03'),
            B04_B = safe_band(SAFE_B, '20190601T103629', 'B04'),
            B08_B = safe_band(SAFE_B, '20190601T103629', 'B08'),
            SCL_B = safe_band(SAFE_B, '20190601T103629', 'SCL', 'R20m'),
            B02_A = safe_band(SAFE_A, '20250828T103559', 'B02'),
            B03_A = safe_band(SAFE_A, '20250828T103559', 'B03'),
            B04_A = safe_band(SAFE_A, '20250828T103559', 'B04'),
            B08_A = safe_band(SAFE_A, '20250828T103559', 'B08'),
            SCL_A = safe_band(SAFE_A, '20250828T103559', 'SCL', 'R20m'),
            worldcover_paths = [p for p in WC_PATHS if os.path.exists(p)] or None,
            output_dir  = str(Path('output') / 'cross_seasonal_2019_2025'),
            date_before = '2019-06-01',
            date_after  = '2025-08-28',
            season_before = 'Late spring (peak greenness)',
            season_after  = 'Late summer (dry season)',
        )

    # ── Same-season pair (new validation pair) ────────────────────────────
    elif pair_name == 'same_season':
        SAFE_B    = 'S2B_MSIL2A_20190601T103629_N0500_R008_T31SGV_20230601T000000.SAFE'
        SAFE_SAME = 'S2B_MSIL2A_20240615T103629_N0511_R008_T31SGV_REPLACE_ME.SAFE'
        DATE_SAME = '20240615T103629'

        return ImagePairConfig(
            name        = 'Same-Season (Jun 2019 vs Jun 2024)',
            description = 'Validation pair — same season eliminates phenological artefact',
            B02_B = safe_band(SAFE_B,    '20190601T103629', 'B02'),
            B03_B = safe_band(SAFE_B,    '20190601T103629', 'B03'),
            B04_B = safe_band(SAFE_B,    '20190601T103629', 'B04'),
            B08_B = safe_band(SAFE_B,    '20190601T103629', 'B08'),
            SCL_B = safe_band(SAFE_B,    '20190601T103629', 'SCL', 'R20m'),
            B02_A = safe_band(SAFE_SAME, DATE_SAME, 'B02'),
            B03_A = safe_band(SAFE_SAME, DATE_SAME, 'B03'),
            B04_A = safe_band(SAFE_SAME, DATE_SAME, 'B04'),
            B08_A = safe_band(SAFE_SAME, DATE_SAME, 'B08'),
            SCL_A = safe_band(SAFE_SAME, DATE_SAME, 'SCL', 'R20m'),
            worldcover_paths = [p for p in WC_PATHS if os.path.exists(p)] or None,
            output_dir  = str(Path('output') / 'same_season_2019_2024'),
            date_before = '2019-06-01',
            date_after  = f'20{DATE_SAME[2:8]}',
            season_before = 'Late spring (peak greenness)',
            season_after  = 'Late spring (same season — validation)',
        )

    else:
        raise ValueError(f"Unknown pair_name: '{pair_name}'. "
                         f"Choose 'cross_seasonal' or 'same_season'.")


# ─────────────────────────────────────────────────────────────────────────────
# SAME-SEASON VALIDATION EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_seasonal_comparison_experiment(cross_pair_config, same_pair_config,
                                        n_estimators=200, max_depth=20,
                                        random_state=42):
    """
    Run the same-season validation experiment.

    Trains the same RF pipeline on BOTH image pairs and compares:
      - RF change rate vs NDVI change rate (per pair)
      - Spatial CV macro-F1 (per pair)
      - NDVI-RF Jaccard (per pair)
      - Feature importance rankings (does NDWI drop for same-season?)
    """
    import pandas as pd
    from algorithms.random_forest import run_random_forest
    from algorithms.ndvi_differencing import run_ndvi_differencing

    results = {}
    for cfg in [cross_pair_config, same_pair_config]:
        ok, missing = cfg.check_files()
        if not ok:
            print(f"\n  SKIPPING {cfg.name} — missing files:")
            for m in missing:
                print(f"    {m}")
            continue

        print(f"\n{'='*60}")
        print(f"Running pipeline for: {cfg.name}")
        print(f"{'='*60}")

        bands_b, bands_a, valid_both = cfg.load_bands_and_mask()

        ndvi_res = run_ndvi_differencing(bands_b[2], bands_b[3],
                                          bands_a[2], bands_a[3])

        rf_res = run_random_forest(
            bands_before=bands_b,
            bands_after=bands_a,
            worldcover_path=cfg.worldcover_paths,
            wc_crop=cfg.crop,
            reference_band_path=cfg.B04_B,
            n_estimators=n_estimators,
            max_depth=max_depth,
            cross_validate=True,
            random_state=random_state,
        )

        results[cfg.name] = {
            'ndvi_change_pct'    : ndvi_res['change_pct'],
            'rf_change_pct'      : rf_res['change_pct'],
            'rf_ndvi_ratio'      : rf_res['change_pct'] / ndvi_res['change_pct'],
            'spatial_cv_f1'      : rf_res.get('spatial_cv_mean'),
            'spatial_cv_std'     : rf_res.get('spatial_cv_std'),
            'holdout_acc'        : rf_res['accuracy'],
            'feature_importance' : rf_res['feature_importance'],
            'ndvi_results'       : ndvi_res,
            'rf_results'         : rf_res,
            'valid_both'         : valid_both,
        }

        ndvi_mask = ndvi_res.get('change_mask')
        rf_mask   = rf_res.get('change_mask')
        if ndvi_mask is not None and rf_mask is not None:
            a = ndvi_mask[valid_both].astype(bool)
            b = rf_mask[valid_both].astype(bool)
            inter = (a & b).sum()
            union = (a | b).sum()
            results[cfg.name]['ndvi_rf_jaccard'] = float(inter/union) \
                                                    if union > 0 else 0.0
        else:
            results[cfg.name]['ndvi_rf_jaccard'] = None

    if not results:
        print("No pairs could be loaded. Check file paths in the configs.")
        return None

    print("\n" + "=" * 80)
    print("SEASONAL COMPARISON EXPERIMENT RESULTS")
    print("=" * 80)

    rows = []
    for pair_name, r in results.items():
        f1_str  = (f"{r['spatial_cv_f1']:.4f} +/- {r['spatial_cv_std']:.4f}"
                   if r['spatial_cv_f1'] is not None else "N/A")
        jac_str = (f"{r['ndvi_rf_jaccard']:.4f}"
                   if r['ndvi_rf_jaccard'] is not None else "N/A")
        rows.append({
            'Pair'             : pair_name,
            'NDVI Change %'    : f"{r['ndvi_change_pct']:.2f}%",
            'RF Change %'      : f"{r['rf_change_pct']:.2f}%",
            'RF/NDVI Ratio'    : f"{r['rf_ndvi_ratio']:.2f}x",
            'Spatial CV F1'    : f1_str,
            'Holdout Acc'      : f"{r['holdout_acc']*100:.1f}%",
            'NDVI-RF Jaccard'  : jac_str,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if len(results) == 2:
        pair_names = list(results.keys())
        fi_cross = results[pair_names[0]].get('feature_importance', {})
        fi_same  = results[pair_names[1]].get('feature_importance', {})

        if fi_cross and fi_same:
            print("\nFEATURE IMPORTANCE COMPARISON:")
            print(f"{'Feature':<16} {'Cross-season':>14} {'Same-season':>13}  Delta")
            print("-" * 55)
            all_feats = sorted(fi_cross, key=lambda f: -fi_cross.get(f, 0))
            for feat in all_feats:
                fc = fi_cross.get(feat, 0)
                fs = fi_same.get(feat, 0)
                marker = " <- NDWI" if feat == 'NDWI' else ""
                print(f"  {feat:<14} {fc*100:>12.2f}%  {fs*100:>12.2f}%  "
                      f"{(fs-fc)*100:>+.2f}pp{marker}")

            ndwi_cross = fi_cross.get('NDWI', 0)
            ndwi_same  = fi_same.get('NDWI', 0)
            delta_ndwi = ndwi_same - ndwi_cross
            print(f"\nNDWI importance: {ndwi_cross*100:.2f}% (cross) "
                  f"-> {ndwi_same*100:.2f}% (same-season)  "
                  f"[{delta_ndwi*100:+.2f} pp]")
            if delta_ndwi < -0.03:
                print("  CONFIRMED: NDWI drops substantially in same-season pair.")
            elif abs(delta_ndwi) < 0.03:
                print("  INCONCLUSIVE: NDWI importance is stable across seasons.")
            else:
                print("  UNEXPECTED: NDWI importance increased for same-season pair.")

    return df, results
