"""
random_forest.py  —  Algorithm 4: Post-Classification Random Forest
====================================================================
Project 23 — LandChangeDetector (Batna Province, Algeria)

Method: post-classification comparison (Project 23 requirement)
  Image_Before → [RF] → LandCover_Before
  Image_After  → [RF] → LandCover_After
  Change Map   = pixels where class differs between dates

Upgrades over v1 (KMeans baseline)
------------------------------------
1. Extended features — 10 features: added SAVI, EVI, BSI
2. Dual-date pooled WorldCover training — trains on 2019+2025 features
   simultaneously to make the RF seasonally robust.
3. Sparse-class guard — merges classes with <MIN_CLASS_SAMPLES pixels.
4. Spatial block CV — geographic train/test split (defensible accuracy).
5. Hyperparameter search — optional grid over max_features × min_leaf.
6. Probability output — classify_image returns confidence map.
7. Confidence-filtered transitions — removes ambiguous-boundary artefacts.
8. Full v1 API compatibility — all existing return dict keys preserved.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

CLASS_LABELS = {
    0: 'Unclassified', 1: 'Vegetation',
    2: 'Bare Soil / Urban', 3: 'Water / Shadow', 4: 'Sparse Vegetation',
}
CLASS_COLORS = {
    0: '#ffffff', 1: '#1a9850', 2: '#d73027',
    3: '#2166ac', 4: '#fee08b',
}
WC_TO_CLASS = {
    10: 1, 20: 4, 30: 4, 40: 1, 50: 2, 60: 2,
    70: 2, 80: 3, 90: 3, 95: 3, 100: 4,
}
MIN_CLASS_SAMPLES = 100
CONF_THRESHOLD    = 0.60
FEATURE_NAMES_V2  = [
    'B02_Blue', 'B03_Green', 'B04_Red', 'B08_NIR',
    'NDVI', 'NDWI', 'Brightness', 'SAVI', 'EVI', 'BSI',
]


# =============================================================================
# 1. EXTENDED FEATURE STACK  (10 features)
# =============================================================================

def build_feature_stack(bands, band_names=None):
    """
    Build the 10-feature spectral matrix for one image date.

    Original 7 features (v1 compatibility):
      B02, B03, B04, B08, NDVI, NDWI, Brightness

    Three new features for semi-arid land classification:
      SAVI  = 1.5*(NIR-Red)/(NIR+Red+0.5)
              Soil-adjusted VI — reduces bright-soil confusion under
              sparse canopy (critical for Batna steppe environment).
      EVI   = 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)
              Enhanced VI — corrects aerosol and background effects.
      BSI   = (Red+Blue-NIR)/(Red+Blue+NIR)
              Bare Soil Index (SWIR-free approximation).
              Positive for bare/urban, negative for vegetation.
              Gives RF a dedicated bare-soil signal without B11.

    Returns
    -------
    features_2d   : np.ndarray float32  (rows*cols, 10)
    feature_names : list[str]
    valid_mask    : np.ndarray bool  (rows, cols)
    """
    blue  = bands[0].astype(np.float32)
    green = bands[1].astype(np.float32)
    red   = bands[2].astype(np.float32)
    nir   = bands[3].astype(np.float32)

    # Original 7
    dv = nir + red
    ndvi = np.where(dv == 0, np.nan, (nir - red) / dv).astype(np.float32)
    dw = green + nir
    ndwi = np.where(dw == 0, np.nan, (green - nir) / dw).astype(np.float32)
    brightness = ((blue + green + red + nir) / 4.0).astype(np.float32)

    # SAVI
    L = 0.5
    ds = nir + red + L
    savi = np.where(ds == 0, np.nan, (1.0 + L) * (nir - red) / ds).astype(np.float32)

    # EVI
    de = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(de == 0, np.nan, 2.5 * (nir - red) / de).astype(np.float32)

    # BSI (SWIR-free)
    db = red + blue + nir
    bsi = np.where(db == 0, np.nan, (red + blue - nir) / db).astype(np.float32)

    stack = np.stack(
        [blue, green, red, nir, ndvi, ndwi, brightness, savi, evi, bsi],
        axis=-1
    )
    valid_mask  = ~np.any(np.isnan(stack), axis=-1)
    rows, cols, nf = stack.shape
    return stack.reshape(-1, nf).astype(np.float32), FEATURE_NAMES_V2, valid_mask


# =============================================================================
# 2. SPARSE-CLASS GUARD
# =============================================================================

def _resolve_active_classes(label_flat, usable_flat,
                             min_samples=MIN_CLASS_SAMPLES):
    """
    Drop / merge classes with insufficient samples.
    Water/Shadow (3) → Bare Soil/Urban (2) when sparse.
    """
    base_classes = [1, 2, 3, 4]
    counts = {c: int((usable_flat & (label_flat == c)).sum())
              for c in base_classes}
    print("\n  Class availability:")
    for c, n in counts.items():
        tag = '✅' if n >= min_samples else f'⚠️  SPARSE (<{min_samples}) → will merge'
        print(f"    Class {c} ({CLASS_LABELS[c]:<20s}): {n:,}  {tag}")

    remap, active = {}, []
    for c in base_classes:
        if counts[c] < min_samples:
            target = 2 if c == 3 else max(
                (x for x in base_classes if x != c and counts[x] >= min_samples),
                key=lambda x: counts[x], default=2
            )
            remap[c] = target
            print(f"  ↳ Merging Class {c} → Class {target}")
        else:
            active.append(c)
    return sorted(active), remap


def _apply_class_remap(labels, remap):
    if not remap:
        return labels
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


# =============================================================================
# 3A. SUPERVISED TRAINING — WorldCover + Dual-Date Pooling
# =============================================================================

def _reproject_worldcover_to_sentinel2(wc_paths, reference_band_path,
                                        wc_crop,
                                        tmp_path='/tmp/wc_aligned.tif'):
    """Reproject WorldCover (EPSG:4326) to Sentinel-2 UTM grid."""
    from osgeo import gdal
    if isinstance(wc_paths, str):
        wc_paths = [wc_paths]

    ref  = gdal.Open(reference_band_path, gdal.GA_ReadOnly)
    proj = ref.GetProjection()
    gt   = ref.GetGeoTransform()
    ref  = None

    xo, yo   = wc_crop['x_off'],  wc_crop['y_off']
    xs, ys   = wc_crop['x_size'], wc_crop['y_size']
    px       = gt[1]
    x_min    = gt[0] + xo * px
    y_max    = gt[3] - yo * px
    x_max, y_min = x_min + xs * px, y_max - ys * px

    if len(wc_paths) > 1:
        print(f"  Mosaicking {len(wc_paths)} WorldCover tiles...")
        vrt = tmp_path.replace('.tif', '.vrt')
        gdal.BuildVRT(vrt, wc_paths)
        src = tmp_path.replace('.tif', '_mosaic.tif')
        gdal.Translate(src, vrt, format='GTiff')
    else:
        src = wc_paths[0]

    print("  Reprojecting WorldCover → UTM...")
    gdal.Warp(tmp_path, src, format='GTiff', dstSRS=proj,
              xRes=px, yRes=px,
              resampleAlg=gdal.GRA_NearestNeighbour,
              outputBounds=(x_min, y_min, x_max, y_max),
              creationOptions=['COMPRESS=LZW'])

    ds   = gdal.Open(tmp_path, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    ds   = None

    if data.shape != (ys, xs):
        from scipy.ndimage import zoom
        data = zoom(data, (ys / data.shape[0], xs / data.shape[1]),
                    order=0).astype(np.uint8)

    print(f"  Aligned: {data.shape}, values: {np.unique(data).tolist()}")
    return data


def generate_supervised_training_worldcover(
        features_before, features_after, valid_mask,
        worldcover_path, wc_crop,
        reference_band_path=None,
        n_samples=50000, random_state=42):
    """
    Generate training samples using ESA WorldCover labels with
    dual-date pooling.

    WHY DUAL-DATE POOLING FIXES THE 61% CHANGE PROBLEM
    ----------------------------------------------------
    v1 trained on 2019 June features only.  The RF learned:
      NDVI=0.25, Red=2800  →  Sparse Vegetation
    In August 2025, the same sparse-veg ground returns:
      NDVI=0.08, Red=4100  →  RF predicts Bare Soil  →  FALSE change

    Dual-date pooling trains on:
      2019 June    sparse-veg pixel  →  Class 4  (WorldCover label)
      2025 August  sparse-veg pixel  →  Class 4  (same WorldCover label)
    Now RF learns that BOTH spectral states mean Class 4.
    The seasonal spectral shift no longer crosses class boundaries.

    Strategy: n_samples/(2*n_classes) pixels per class per date.
    Samples are spatially non-overlapping between dates.
    """
    # Load WorldCover
    if reference_band_path is not None:
        wc_data = _reproject_worldcover_to_sentinel2(
            wc_paths=worldcover_path,
            reference_band_path=reference_band_path,
            wc_crop=wc_crop,
        )
    else:
        try:
            from ..utils.raster_utils import load_band_crop
        except ImportError:
            from utils.raster_utils import load_band_crop
        paths = ([worldcover_path] if isinstance(worldcover_path, str)
                 else worldcover_path)
        wc_data, _, _ = load_band_crop(paths[0], **wc_crop)
        wc_data = wc_data.astype(np.uint8)

    # Map WC → project classes
    label_map = np.zeros(wc_data.shape, dtype=np.uint8)
    for wv, cls in WC_TO_CLASS.items():
        label_map[wc_data == wv] = cls

    usable      = valid_mask & (label_map > 0)
    label_flat  = label_map.ravel()
    usable_flat = usable.ravel()
    print(f"  WC coverage: {(label_map > 0).mean()*100:.1f}%  |  "
          f"usable intersection: {usable.sum():,}")

    # Sparse-class guard
    active_classes, remap = _resolve_active_classes(label_flat, usable_flat)
    if remap:
        label_flat = _apply_class_remap(label_flat, remap)

    # Stratified dual-date sampling
    per_class      = n_samples // len(active_classes)
    per_class_date = per_class // 2
    rng            = np.random.RandomState(random_state)
    X_parts, y_parts, idx_parts = [], [], []
    label_counts = {}

    print(f"\n  Dual-date sampling ({per_class_date}/class/date):")
    for cls in active_classes:
        indices = np.where(usable_flat & (label_flat == cls))[0]
        n_b     = min(per_class_date, len(indices))
        idx_b   = rng.choice(indices, n_b, replace=False)

        remaining = indices[~np.isin(indices, idx_b)]
        n_a       = min(per_class_date, len(remaining))
        idx_a     = (rng.choice(remaining, n_a, replace=False)
                     if n_a > 0 else np.array([], dtype=np.int64))

        for feat, idx, tag in [(features_before, idx_b, '2019'),
                                (features_after,  idx_a, '2025')]:
            if len(idx) == 0:
                continue
            X_parts.append(feat[idx])
            y_parts.append(np.full(len(idx), cls, dtype=np.uint8))
            idx_parts.append(idx)

        label_counts[cls] = n_b + n_a
        print(f"    Class {cls} ({CLASS_LABELS[cls]:<20s}): "
              f"{n_b}(2019)+{n_a}(2025)={n_b+n_a}")

    X_train  = np.vstack(X_parts)
    y_train  = np.concatenate(y_parts)
    all_idx  = np.concatenate(idx_parts)

    shuf    = rng.permutation(len(X_train))
    X_train = X_train[shuf]
    y_train = y_train[shuf]
    all_idx = all_idx[shuf]

    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"\n  Total: {len(X_train):,} samples")
    return X_train, y_train, scaler, label_counts, remap, all_idx


# =============================================================================
# 3B. UNSUPERVISED FALLBACK — KMeans  (v1 preserved for compatibility)
# =============================================================================

def generate_training_samples(features_2d, valid_mask,
                               n_classes=4, n_samples=5000,
                               random_state=42):
    """KMeans pseudo-label fallback. Use only when WorldCover unavailable."""
    valid_flat = valid_mask.flatten()
    valid_feat = features_2d[valid_flat]
    print(f"  Valid pixels: {valid_feat.shape[0]:,}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid_feat)
    n_cl   = min(50000, valid_feat.shape[0])
    rng    = np.random.RandomState(random_state)
    idx    = rng.choice(valid_feat.shape[0], n_cl, replace=False)
    print(f"  KMeans ({n_classes} clusters, {n_cl:,} px)...")
    km = KMeans(n_clusters=n_classes, random_state=random_state,
                n_init=10, max_iter=300)
    km.fit(scaled[idx])
    all_labels = km.predict(scaled) + 1
    n_tr       = min(n_samples, len(all_labels))
    tr_idx     = rng.choice(len(all_labels), n_tr, replace=False)
    X_tr, y_tr = valid_feat[tr_idx], all_labels[tr_idx]
    flat_idx   = np.where(valid_flat)[0][tr_idx]
    lc         = {int(u): int(c)
                  for u, c in zip(*np.unique(y_tr, return_counts=True))}
    for cls, cnt in lc.items():
        print(f"    Class {cls} ({CLASS_LABELS.get(cls,''):<18s}): {cnt:,}")
    return X_tr, y_tr, km, scaler, lc, flat_idx


# =============================================================================
# 4. SPATIAL BLOCK CROSS-VALIDATION
# =============================================================================

class SpatialBlockSplit:
    """
    Spatial block CV splitter.

    Divides the 2D crop into n_blocks_r * n_blocks_c geographic blocks.
    Each fold holds out ONE block as test; trains on all others.
    Adjacent pixels are always in the SAME set, eliminating spatial
    autocorrelation leakage between train/test.

    This produces the only defensible accuracy figure for RS data.
    Standard random splits inflate accuracy by 5-15 pp because test
    pixels are neighbours of training pixels.
    """
    def __init__(self, pixel_flat_indices, image_shape,
                 n_blocks_r=4, n_blocks_c=4):
        rows, cols = image_shape
        pr = pixel_flat_indices // cols
        pc = pixel_flat_indices %  cols
        bh = rows // n_blocks_r
        bw = cols // n_blocks_c
        br = np.minimum(pr // bh, n_blocks_r - 1)
        bc = np.minimum(pc // bw, n_blocks_c - 1)
        self.block_ids  = br * n_blocks_c + bc
        self.n_splits   = n_blocks_r * n_blocks_c
        u, cnt = np.unique(self.block_ids, return_counts=True)
        print(f"  Spatial CV: {n_blocks_r}×{n_blocks_c}={self.n_splits} blocks  "
              f"(min/max samples: {cnt.min()}/{cnt.max()})")

    def split(self):
        for blk in np.unique(self.block_ids):
            te = (self.block_ids == blk)
            tr = ~te
            if te.sum() > 0 and tr.sum() > 0:
                yield np.where(tr)[0], np.where(te)[0]


def _spatial_cv_score(clf_params, X_scaled, y, splitter):
    """Return mean macro-F1 over spatial folds."""
    # Avoid duplicate kwargs when caller already provides n_jobs.
    rf_params = dict(clf_params)
    rf_params.setdefault('n_jobs', -1)

    scores = []
    for tr, te in splitter.split():
        if len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_scaled[tr], y[tr])
        scores.append(f1_score(y[te], clf.predict(X_scaled[te]),
                               average='macro', zero_division=0))
    if not scores:
        return 0.0, 0.0, []
    return float(np.mean(scores)), float(np.std(scores)), scores


# =============================================================================
# 5. RF TRAINING WITH SPATIAL CV AND OPTIONAL HYPERPARAMETER SEARCH
# =============================================================================

def train_random_forest(X_train, y_train, scaler,
                        pixel_flat_indices, image_shape,
                        n_estimators=200, max_depth=20,
                        optimize_hyperparams=False,
                        cross_validate=True,
                        random_state=42):
    """
    Train RF with spatial block cross-validation.

    If optimize_hyperparams=True: runs a 9-config grid search over
      max_features ∈ {sqrt, 0.3, 0.4} × min_samples_leaf ∈ {3, 5, 10}
    scored with spatial CV macro-F1.

    Reports TWO accuracy numbers:
      • Hold-out accuracy (random split) — kept for v1 summary-table compat.
      • Spatial CV macro-F1 ± std         — the defensible thesis figure.

    Returns
    -------
    clf, holdout_acc, spatial_cv_mean, spatial_cv_std, feat_importance, best_params
    """
    X_sc = scaler.transform(X_train)
    X_tr, X_val, y_tr, y_val, idx_tr, _ = train_test_split(
        X_sc, y_train, pixel_flat_indices,
        test_size=0.2, random_state=random_state, stratify=y_train,
    )

    splitter = SpatialBlockSplit(pixel_flat_indices, image_shape)

    # Hyperparameter search
    if optimize_hyperparams:
        grid = list(ParameterGrid({
            'max_features':      ['sqrt', 0.3, 0.4],
            'min_samples_leaf':  [3, 5, 10],
        }))
        print(f"\n  Hyperparameter search: {len(grid)} configs × "
              f"{splitter.n_splits} folds...")
        best_f1, best_params = -1.0, grid[0]
        for p in grid:
            base = dict(n_estimators=n_estimators, max_depth=max_depth,
                        class_weight='balanced', oob_score=False,
                        random_state=random_state, **p)
            mf1, sf1, _ = _spatial_cv_score(base, X_sc, y_train, splitter)
            print(f"    {p}  →  F1={mf1:.4f}±{sf1:.4f}")
            if mf1 > best_f1:
                best_f1, best_params = mf1, p
        print(f"  Best: {best_params}  (F1={best_f1:.4f})")
    else:
        best_params = {'max_features': 0.4, 'min_samples_leaf': 5}

    # Train final model
    final_params = dict(n_estimators=n_estimators, max_depth=max_depth,
                        class_weight='balanced', oob_score=True, n_jobs=-1,
                        random_state=random_state, **best_params)
    print(f"\n  Training RF  (trees={n_estimators}, depth={max_depth}, "
          f"max_feat={best_params['max_features']}, "
          f"min_leaf={best_params['min_samples_leaf']})")
    print(f"  Train: {len(X_tr):,}  Val(random): {len(X_val):,}")

    clf = RandomForestClassifier(**final_params)
    clf.fit(X_tr, y_tr)

    holdout_acc = float((clf.predict(X_val) == y_val).mean())
    print(f"  Hold-out accuracy   : {holdout_acc*100:.1f}%  "
          f"[random split — optimistic, do NOT report as primary]")
    if hasattr(clf, 'oob_score_'):
        print(f"  OOB accuracy        : {clf.oob_score_*100:.1f}%")

    # Per-class report
    cids  = sorted(np.unique(y_train))
    clbls = [CLASS_LABELS.get(c, f'C{c}') for c in cids]
    print("\n  Per-class (random hold-out):")
    for ln in classification_report(y_val, clf.predict(X_val),
                                     labels=cids, target_names=clbls,
                                     zero_division=0).splitlines():
        print(f"    {ln}")

    # Spatial CV on full dataset
    spatial_cv_mean = spatial_cv_std = None
    if cross_validate:
        print(f"\n  Spatial block CV ({splitter.n_splits} folds)...")
        clf_full = RandomForestClassifier(**final_params)
        clf_full.fit(X_sc, y_train)
        mf1, sf1, folds = _spatial_cv_score(
            {**final_params, 'oob_score': False},
            X_sc, y_train, splitter
        )
        spatial_cv_mean = mf1
        spatial_cv_std  = sf1
        print(f"  Spatial CV macro-F1 : {mf1:.4f} ± {sf1:.4f}  ← REPORT THIS")
        print(f"  Per-fold F1         : {[f'{s:.3f}' for s in folds]}")
        clf = clf_full   # use full-data model for classification

    # Feature importance
    fnames = FEATURE_NAMES_V2[:clf.n_features_in_]
    fimp   = {n: float(i) for n, i in zip(fnames, clf.feature_importances_)}
    print("\n  Feature importances:")
    for n, i in sorted(fimp.items(), key=lambda x: -x[1]):
        print(f"    {n:<14s} {i:.4f}  {'█'*int(i*40)}")

    return clf, holdout_acc, spatial_cv_mean, spatial_cv_std, fimp, best_params


# =============================================================================
# 6. CLASSIFICATION WITH PROBABILITY OUTPUT
# =============================================================================

def classify_image(clf, features_2d, valid_mask, image_shape, scaler):
    """
    Classify all valid pixels; return hard class map AND confidence map.

    Confidence = max class probability from RF.
    Used for confidence-filtered transition matrix.

    Returns
    -------
    class_map : uint8   (rows, cols)  0=masked
    conf_map  : float32 (rows, cols)  0.0–1.0
    """
    rows, cols = image_shape
    class_flat = np.zeros(rows * cols, dtype=np.uint8)
    conf_flat  = np.zeros(rows * cols, dtype=np.float32)
    vf         = valid_mask.flatten()
    vx         = scaler.transform(features_2d[vf])

    hards, probs = [], []
    for i in range(0, len(vx), 100_000):
        b = vx[i:i+100_000]
        hards.append(clf.predict(b))
        probs.append(clf.predict_proba(b).max(axis=1))
        print(f"  Classifying... {min(100,(i+100000)/len(vx)*100):.0f}%",
              end='\r')
    print()

    class_flat[vf] = np.concatenate(hards)
    conf_flat[vf]  = np.concatenate(probs)
    return (class_flat.reshape(rows, cols).astype(np.uint8),
            conf_flat.reshape(rows, cols).astype(np.float32))


# =============================================================================
# 7. CONFIDENCE-FILTERED TRANSITION MATRIX
# =============================================================================

def compute_transition_matrix(map_before, map_after,
                               conf_before=None, conf_after=None,
                               n_classes=4,
                               conf_threshold=CONF_THRESHOLD):
    """
    Standard + confidence-filtered transition matrices.

    Standard: counts all valid pixels (v1 behaviour).
    Confident: counts only pixels where BOTH dates have
               RF confidence >= conf_threshold.  Removes
               low-confidence boundary artefacts from the
               transition table.

    Returns
    -------
    matrix, matrix_pct,
    matrix_conf (or None), matrix_conf_pct (or None),
    n_confident_pixels
    """
    cls = list(range(1, n_classes + 1))
    n   = len(cls)
    v   = (map_before > 0) & (map_after > 0)

    def _mat(mask):
        m = np.zeros((n, n), dtype=np.int64)
        for i, ci in enumerate(cls):
            for j, cj in enumerate(cls):
                m[i, j] = int(((map_before==ci) & (map_after==cj) & mask).sum())
        return m

    def _pct(m):
        rs = m.sum(axis=1, keepdims=True).astype(float)
        rs[rs == 0] = 1.0
        return (m / rs) * 100.0

    mat = _mat(v)
    mat_pct = _pct(mat)

    mat_c = mat_c_pct = None
    n_conf = 0
    if conf_before is not None and conf_after is not None:
        cm   = (conf_before >= conf_threshold) & \
               (conf_after  >= conf_threshold) & v
        n_conf = int(cm.sum())
        if n_conf > 0:
            mat_c     = _mat(cm)
            mat_c_pct = _pct(mat_c)

    return mat, mat_pct, mat_c, mat_c_pct, n_conf


# =============================================================================
# 8. MODEL PERSISTENCE
# =============================================================================

def save_classifier(clf, scaler, output_path):
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    joblib.dump({'clf': clf, 'scaler': scaler}, output_path, compress=3)
    print(f"  Saved → {output_path}")


def load_classifier(model_path):
    b = joblib.load(model_path)
    print(f"  Loaded: {model_path}  (trees={b['clf'].n_estimators})")
    return b['clf'], b['scaler']


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def run_random_forest(bands_before, bands_after,
                      n_classes=4,
                      n_train_samples=50000,
                      worldcover_path=None,
                      wc_crop=None,
                      reference_band_path=None,
                      n_estimators=200,
                      max_depth=20,
                      optimize_hyperparams=False,
                      cross_validate=True,
                      conf_threshold=CONF_THRESHOLD,
                      save_model_path=None,
                      random_state=42):
    """
    Post-classification RF change detection — upgraded v2 pipeline.
    Full v1 return-dict API preserved.  New keys added.
    """
    rows, cols  = bands_before[0].shape
    image_shape = (rows, cols)

    print("=" * 60)
    print("STEP 1: Building extended feature stacks (10 features)")
    feat_b, fnames, val_b = build_feature_stack(bands_before)
    feat_a, _,     val_a = build_feature_stack(bands_after)
    valid_both = val_b & val_a
    print(f"  Valid pixels (both dates): {valid_both.sum():,}")

    print("\nSTEP 2: Generating training samples")
    use_wc = (worldcover_path is not None) and (wc_crop is not None)

    if use_wc:
        training_mode = 'supervised'
        print("  Mode: SUPERVISED — dual-date pooled WorldCover 2021")
        (X_train, y_train, scaler,
         label_counts, remap, samp_idx) = \
            generate_supervised_training_worldcover(
                feat_b, feat_a, valid_both,
                worldcover_path=worldcover_path,
                wc_crop=wc_crop,
                reference_band_path=reference_band_path,
                n_samples=n_train_samples,
                random_state=random_state,
            )
    else:
        training_mode = 'unsupervised'
        print("  Mode: UNSUPERVISED (KMeans fallback)")
        print("  ⚠️  Expect ~50% inflated change (seasonal mismatch)")
        cap = min(n_train_samples, 5000)
        (X_train, y_train, _km, scaler,
         label_counts, samp_idx) = \
            generate_training_samples(
                feat_b, valid_both,
                n_classes=n_classes, n_samples=cap,
                random_state=random_state,
            )
        remap = {}

    print("\nSTEP 3: Training Random Forest")
    (clf, holdout_acc, sp_mean, sp_std,
     fimp, best_params) = train_random_forest(
        X_train, y_train, scaler,
        pixel_flat_indices=samp_idx,
        image_shape=image_shape,
        n_estimators=n_estimators, max_depth=max_depth,
        optimize_hyperparams=optimize_hyperparams,
        cross_validate=cross_validate,
        random_state=random_state,
    )

    print("\nSTEP 4: Classifying BEFORE image (2019)")
    map_b, prob_b = classify_image(clf, feat_b, valid_both, image_shape, scaler)
    print("STEP 5: Classifying AFTER image (2025)")
    map_a, prob_a = classify_image(clf, feat_a, valid_both, image_shape, scaler)

    change_map = np.where(valid_both, (map_b != map_a).astype(np.uint8), 0)
    n_chg      = int(change_map[valid_both].sum())
    n_val      = int(valid_both.sum())
    chg_pct    = (n_chg / n_val * 100.0) if n_val > 0 else 0.0

    print("\nSTEP 6: Computing transition matrices")
    mat, mat_pct, mat_c, mat_c_pct, n_conf = compute_transition_matrix(
        map_b, map_a,
        conf_before=prob_b, conf_after=prob_a,
        n_classes=n_classes, conf_threshold=conf_threshold,
    )

    print(f"\n{'─'*60}")
    print(f"  Changed pixels    : {n_chg:,} ({chg_pct:.1f}%)")
    print(f"  Hold-out accuracy : {holdout_acc*100:.1f}%  [random split]")
    if sp_mean is not None:
        print(f"  Spatial CV F1     : {sp_mean:.4f}±{sp_std:.4f}  ← cite this")
    if n_conf > 0:
        print(f"  Confident pixels  : {n_conf:,} "
              f"({n_conf/n_val*100:.1f}%, conf≥{conf_threshold})")
    print(f"  Training mode     : {training_mode}")
    print(f"{'─'*60}")

    if save_model_path:
        save_classifier(clf, scaler, save_model_path)

    class_names = [CLASS_LABELS.get(i, f'Class {i}')
                   for i in range(1, n_classes + 1)]

    return {
        # v1 keys (ALL preserved for full compatibility)
        'map_before'                 : map_b,
        'map_after'                  : map_a,
        'change_map'                 : change_map,
        'class_before'               : map_b,
        'class_after'                : map_a,
        'change_mask'                : change_map.astype(bool),
        'transition_matrix'          : mat,
        'transition_matrix_pct'      : mat_pct,
        'change_pct'                 : chg_pct,
        'accuracy'                   : holdout_acc,
        'cv_accuracy'                : holdout_acc,
        'clf'                        : clf,
        'scaler'                     : scaler,
        'valid_mask'                 : valid_both,
        'feature_names'              : fnames,
        'feature_importance'         : fimp,
        'class_names'                : class_names,
        'training_mode'              : training_mode,
        'label_counts'               : label_counts,
        # New v2 keys
        'prob_before'                : prob_b,
        'prob_after'                 : prob_a,
        'transition_matrix_conf'     : mat_c,
        'transition_matrix_conf_pct' : mat_c_pct,
        'n_confident_pixels'         : n_conf,
        'spatial_cv_mean'            : sp_mean,
        'spatial_cv_std'             : sp_std,
        'best_hyperparams'           : best_params,
        'class_remap'                : remap,
        'n_features'                 : len(fnames),
        'conf_threshold'             : conf_threshold,
    }
