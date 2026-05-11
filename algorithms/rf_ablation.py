"""
algorithms/rf_ablation.py
=========================
Project 23 — LandChangeDetector  (Batna Province, Algeria)

NDWI Ablation Study for the Random Forest classifier.

Purpose
-------
Test whether NDWI (Normalised Difference Water Index) is helping or
hurting the RF classifier for cross-seasonal Sentinel-2 imagery.

NDWI = (Green - NIR) / (Green + NIR)

Concern: NDWI is the top feature at 21.34% importance in the current
10-feature RF. In June 2019, active vegetation transpiration elevates
Green reflectance relative to NIR. In August 2025, the dry landscape
has lower atmospheric moisture. NDWI therefore shifts substantially
between the two seasons, potentially encoding the seasonal spectral
shift as a pseudo-feature rather than genuine land-cover information.

Experiment Design
-----------------
Run N+1 RF models:
  - Baseline:    all 10 features (B02,B03,B04,B08,NDVI,NDWI,Brt,SAVI,EVI,BSI)
  - No-NDWI:     9 features (NDWI removed)
  - No-EVI:      9 features (EVI removed — control experiment)
  - No-SAVI:     9 features (SAVI removed — control)
  - No-BSI:      9 features (BSI removed — control)
  - No-NDVI:     9 features (NDVI removed — sanity check; should hurt badly)
  - Veg-only:    {NDVI,SAVI,EVI} only — minimal vegetation-specific set
  - Spectral-only: {B02,B03,B04,B08} only — raw bands, no derived indices

For each model record:
  1. Spatial block CV macro-F1 (primary metric)
  2. NDVI–RF Jaccard (secondary metric — spatial alignment with NDVI)
  3. RF change rate % (sanity check)
  4. Hold-out accuracy (for summary table only — do not cite as primary)

Decision rule
-------------
  NDWI is a contaminating feature IF:
    removing NDWI → spatial CV F1 improves by > 0.02
    OR  removing NDWI → NDVI-RF Jaccard improves by > 0.05

  NDWI is genuinely informative IF:
    removing NDWI → spatial CV F1 drops by > 0.02
    AND  removing NDWI → Jaccard does NOT improve

  Results are ambiguous IF:
    changes are within ±0.02 F1 / ±0.05 Jaccard

Usage
-----
    # In notebook after rf_results is available:
    from algorithms.rf_ablation import run_ablation_study
    ablation_results = run_ablation_study(
        bands_before = [blue_b_m, green_b_m, red_b_m, nir_b_m],
        bands_after  = [blue_a_m, green_a_m, red_a_m, nir_a_m],
        valid_both   = valid_both,
        worldcover_path      = WORLDCOVER_PATHS,
        wc_crop              = CROP,
        reference_band_path  = B04_B,
        ndvi_results         = ndvi_results,
    )
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from algorithms.random_forest import (
    build_feature_stack,
    generate_supervised_training_worldcover,
    generate_training_samples,
    SpatialBlockSplit,
    _spatial_cv_score,
    classify_image,
    FEATURE_NAMES_V2,
)


# ── Feature index map ─────────────────────────────────────────────────────
# Matches the order in build_feature_stack() output
FEAT_IDX = {
    'B02_Blue'   : 0,
    'B03_Green'  : 1,
    'B04_Red'    : 2,
    'B08_NIR'    : 3,
    'NDVI'       : 4,
    'NDWI'       : 5,
    'Brightness' : 6,
    'SAVI'       : 7,
    'EVI'        : 8,
    'BSI'        : 9,
}

# ── Experiment configurations ─────────────────────────────────────────────
EXPERIMENTS = {
    'Baseline (all 10)'   : list(range(10)),          # all features
    'No NDWI'             : [0,1,2,3,4,6,7,8,9],     # drop NDWI (idx 5)
    'No EVI'              : [0,1,2,3,4,5,6,7,9],     # drop EVI  (idx 8)
    'No SAVI'             : [0,1,2,3,4,5,6,8,9],     # drop SAVI (idx 7)
    'No BSI'              : [0,1,2,3,4,5,6,7,8],     # drop BSI  (idx 9)
    'No NDVI'             : [0,1,2,3,5,6,7,8,9],     # drop NDVI (idx 4)
    'No Brightness'       : [0,1,2,3,4,5,7,8,9],     # drop Brightness
    'Veg indices only'    : [4,5,7,8],                # NDVI,NDWI,SAVI,EVI
    'Spectral bands only' : [0,1,2,3],                # B02,B03,B04,B08
    'No moisture indices' : [0,1,2,3,4,6,7,8,9],     # No NDWI (same as No NDWI — alias)
}
# Note: 'No NDWI' and 'No moisture indices' are identical by design —
# the duplicate is removed in run_ablation_study before execution.


def _compute_jaccard(rf_change_mask, ndvi_change_mask, valid_mask):
    """Compute Jaccard similarity between RF and NDVI change masks."""
    a = rf_change_mask[valid_mask].astype(bool)
    b = ndvi_change_mask[valid_mask].astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def run_single_experiment(name, feature_indices,
                           X_train_full, y_train, pixel_flat_indices,
                           features_before_full, features_after_full,
                           valid_both, image_shape,
                           ndvi_change_mask,
                           n_estimators=200, max_depth=20,
                           random_state=42):
    """
    Run one RF experiment with a specific feature subset.

    Parameters
    ----------
    name              : str   experiment name (for display)
    feature_indices   : list  column indices to select from the 10-feature matrix
    X_train_full      : np.ndarray  full 10-feature training matrix
    y_train           : np.ndarray  training labels
    pixel_flat_indices: np.ndarray  flat pixel positions of training samples
    features_before_full : np.ndarray  (n_pixels, 10) before image features
    features_after_full  : np.ndarray  (n_pixels, 10) after image features
    valid_both        : np.ndarray  bool  (rows, cols)
    image_shape       : (rows, cols)
    ndvi_change_mask  : np.ndarray  bool  NDVI change mask for Jaccard

    Returns
    -------
    dict  with keys: name, n_features, features_used, spatial_cv_f1,
                     spatial_cv_std, holdout_acc, rf_change_pct,
                     ndvi_rf_jaccard, fold_f1s
    """
    feat_names = [FEATURE_NAMES_V2[i] for i in feature_indices]
    print(f"\n  [{name}]  features: {feat_names}")

    # Subset features
    X_tr = X_train_full[:, feature_indices]
    X_be = features_before_full[:, feature_indices]
    X_af = features_after_full[:, feature_indices]

    # Scale
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)

    # Spatial block CV
    splitter = SpatialBlockSplit(pixel_flat_indices, image_shape,
                                  n_blocks_r=4, n_blocks_c=4)
    rf_params = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight='balanced', oob_score=False,
        random_state=random_state,
        max_features=0.4, min_samples_leaf=5,
    )
    sp_f1, sp_std, fold_f1s = _spatial_cv_score(rf_params, X_tr_sc,
                                                  y_train, splitter)
    print(f"    Spatial CV macro-F1: {sp_f1:.4f} ± {sp_std:.4f}")

    # Train final model on all training data
    clf = RandomForestClassifier(**rf_params, n_jobs=-1)
    clf.fit(X_tr_sc, y_train)

    # Hold-out accuracy (random split — for reference only)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr_sc, y_train, test_size=0.2,
        random_state=random_state, stratify=y_train
    )
    clf2 = RandomForestClassifier(**rf_params, n_jobs=-1)
    clf2.fit(X_tr2, y_tr2)
    holdout_acc = float((clf2.predict(X_val) == y_val).mean())

    # Classify both images
    map_b, _ = classify_image(clf, X_be, valid_both, image_shape, scaler)
    map_a, _ = classify_image(clf, X_af, valid_both, image_shape, scaler)

    change_map = np.where(valid_both,
                          (map_b != map_a).astype(np.uint8), 0)
    n_valid    = int(valid_both.sum())
    rf_chg_pct = float(change_map[valid_both].sum() / n_valid * 100)
    jaccard    = _compute_jaccard(change_map.astype(bool),
                                   ndvi_change_mask,
                                   valid_both)

    print(f"    Hold-out acc  : {holdout_acc*100:.1f}%")
    print(f"    RF change %   : {rf_chg_pct:.2f}%")
    print(f"    NDVI-RF Jaccard: {jaccard:.4f}")

    return {
        'name'           : name,
        'n_features'     : len(feature_indices),
        'features_used'  : feat_names,
        'spatial_cv_f1'  : sp_f1,
        'spatial_cv_std' : sp_std,
        'holdout_acc'    : holdout_acc,
        'rf_change_pct'  : rf_chg_pct,
        'ndvi_rf_jaccard': jaccard,
        'fold_f1s'       : fold_f1s,
    }


def run_ablation_study(bands_before, bands_after, valid_both,
                        ndvi_results,
                        worldcover_path=None, wc_crop=None,
                        reference_band_path=None,
                        n_train_samples=50000,
                        n_estimators=200, max_depth=20,
                        random_state=42):
    """
    Run the complete NDWI ablation study.

    Trains one RF model per experiment configuration (10 configs),
    records spatial CV macro-F1, NDVI-RF Jaccard, and RF change rate
    for each, then returns a comparison DataFrame.

    Parameters
    ----------
    bands_before, bands_after : lists of [B02,B03,B04,B08] np.ndarray
    valid_both        : bool ndarray  combined cloud-free mask
    ndvi_results      : dict  from run_ndvi_differencing() — for Jaccard
    worldcover_path   : str | list | None  for supervised training
    wc_crop           : dict  crop window
    reference_band_path : str  for WorldCover reprojection
    n_train_samples   : int
    n_estimators, max_depth, random_state : RF hyperparameters

    Returns
    -------
    results_df  : pd.DataFrame  one row per experiment
    results_raw : list[dict]    raw result dicts
    conclusion  : str           human-readable interpretation
    """
    rows, cols  = bands_before[0].shape
    image_shape = (rows, cols)

    print("=" * 60)
    print("NDWI ABLATION STUDY")
    print(f"  10-feature baseline vs 9 alternative configurations")
    print("=" * 60)

    # ── Step 1: Build full 10-feature stacks ─────────────────────────────────
    print("\nStep 1: Building 10-feature stacks...")
    from algorithms.random_forest import build_feature_stack
    feat_b_full, _, _ = build_feature_stack(bands_before)
    feat_a_full, _, _ = build_feature_stack(bands_after)

    # ── Step 2: Generate training samples ONCE (all experiments reuse them) ──
    print("\nStep 2: Generating training samples (done once for all experiments)...")
    use_wc = (worldcover_path is not None) and (wc_crop is not None)

    if use_wc:
        print("  Training mode: SUPERVISED (WorldCover)")
        (X_train_full, y_train, scaler_full,
         label_counts, remap, samp_idx) = \
            generate_supervised_training_worldcover(
                feat_b_full, feat_a_full, valid_both,
                worldcover_path=worldcover_path,
                wc_crop=wc_crop,
                reference_band_path=reference_band_path,
                n_samples=n_train_samples,
                random_state=random_state,
            )
    else:
        print("  Training mode: UNSUPERVISED (KMeans fallback)")
        cap = min(n_train_samples, 5000)
        (X_train_full, y_train, _km, scaler_full,
         label_counts, samp_idx) = \
            generate_training_samples(
                feat_b_full, valid_both,
                n_classes=4, n_samples=cap,
                random_state=random_state,
            )

    print(f"  Training set: {len(X_train_full):,} samples, "
          f"{X_train_full.shape[1]} features")

    # NDVI change mask for Jaccard computation
    ndvi_change_mask = ndvi_results.get('change_mask',
                       ndvi_results.get('change_map', None))
    if ndvi_change_mask is None:
        # Reconstruct from loss+gain masks
        ndvi_change_mask = (ndvi_results.get('loss_mask', np.zeros(image_shape, bool)) |
                            ndvi_results.get('gain_mask', np.zeros(image_shape, bool)))

    # ── Step 3: Run all experiments ───────────────────────────────────────────
    print("\nStep 3: Running ablation experiments...")

    # De-duplicate (No NDWI and No moisture indices are identical)
    seen        = {}
    unique_exps = {}
    for name, indices in EXPERIMENTS.items():
        key = tuple(sorted(indices))
        if key not in seen:
            seen[key]          = name
            unique_exps[name]  = indices
        else:
            print(f"  [skip duplicate] '{name}' == '{seen[key]}'")

    raw_results = []
    for i, (name, indices) in enumerate(unique_exps.items()):
        print(f"\n  Experiment {i+1}/{len(unique_exps)}: {name}")
        res = run_single_experiment(
            name=name,
            feature_indices=indices,
            X_train_full=X_train_full,
            y_train=y_train,
            pixel_flat_indices=samp_idx,
            features_before_full=feat_b_full,
            features_after_full=feat_a_full,
            valid_both=valid_both,
            image_shape=image_shape,
            ndvi_change_mask=ndvi_change_mask,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        raw_results.append(res)

    # ── Step 4: Assemble comparison DataFrame ─────────────────────────────────
    baseline = next(r for r in raw_results if r['name'] == 'Baseline (all 10)')

    rows_df = []
    for r in raw_results:
        delta_f1  = r['spatial_cv_f1'] - baseline['spatial_cv_f1']
        delta_jac = r['ndvi_rf_jaccard'] - baseline['ndvi_rf_jaccard']
        delta_chg = r['rf_change_pct']  - baseline['rf_change_pct']

        if r['name'] == 'Baseline (all 10)':
            verdict = 'BASELINE'
        elif delta_f1 > 0.02 or delta_jac > 0.05:
            verdict = 'BETTER (removing helps)'
        elif delta_f1 < -0.02:
            verdict = 'WORSE (removing hurts)'
        else:
            verdict = 'NEUTRAL (±noise)'

        rows_df.append({
            'Experiment'         : r['name'],
            'N Features'         : r['n_features'],
            'Spatial CV F1'      : f"{r['spatial_cv_f1']:.4f}",
            'CV Std'             : f"{r['spatial_cv_std']:.4f}",
            'Delta F1'           : f"{delta_f1:+.4f}",
            'Holdout Acc'        : f"{r['holdout_acc']*100:.1f}%",
            'RF Change %'        : f"{r['rf_change_pct']:.2f}%",
            'Delta Change'       : f"{delta_chg:+.2f}pp",
            'NDVI-RF Jaccard'    : f"{r['ndvi_rf_jaccard']:.4f}",
            'Delta Jaccard'      : f"{delta_jac:+.4f}",
            'Verdict'            : verdict,
        })

    df = pd.DataFrame(rows_df)

    # ── Step 5: Interpret results ─────────────────────────────────────────────
    no_ndwi_row = next((r for r in raw_results if r['name'] == 'No NDWI'), None)
    conclusion  = _interpret_ndwi_result(baseline, no_ndwi_row)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df.to_string(index=False))
    print()
    print("NDWI VERDICT:")
    print(conclusion)
    print("=" * 60)

    return df, raw_results, conclusion


def _interpret_ndwi_result(baseline, no_ndwi):
    """Generate a human-readable interpretation of the NDWI ablation."""
    if no_ndwi is None:
        return "No NDWI experiment found in results."

    delta_f1  = no_ndwi['spatial_cv_f1']  - baseline['spatial_cv_f1']
    delta_jac = no_ndwi['ndvi_rf_jaccard'] - baseline['ndvi_rf_jaccard']
    delta_chg = no_ndwi['rf_change_pct']   - baseline['rf_change_pct']

    lines = [
        f"Baseline (with NDWI) : spatial CV F1 = {baseline['spatial_cv_f1']:.4f}, "
        f"Jaccard = {baseline['ndvi_rf_jaccard']:.4f}, "
        f"Change = {baseline['rf_change_pct']:.2f}%",
        f"No NDWI              : spatial CV F1 = {no_ndwi['spatial_cv_f1']:.4f}, "
        f"Jaccard = {no_ndwi['ndvi_rf_jaccard']:.4f}, "
        f"Change = {no_ndwi['rf_change_pct']:.2f}%",
        f"Delta F1    : {delta_f1:+.4f}",
        f"Delta Jaccard: {delta_jac:+.4f}",
        f"Delta Change : {delta_chg:+.2f} pp",
        "",
    ]

    if delta_f1 > 0.02 or delta_jac > 0.05:
        lines += [
            "VERDICT: NDWI IS A CONTAMINATING FEATURE",
            "Removing NDWI IMPROVES classification performance.",
            "NDWI was encoding the June-August seasonal moisture shift",
            "as a pseudo-feature rather than genuine land-cover information.",
            "ACTION: Remove NDWI from the feature set for cross-seasonal",
            "        applications. Retain it for same-season pairs where",
            "        moisture differences carry real class information.",
        ]
    elif delta_f1 < -0.02:
        lines += [
            "VERDICT: NDWI IS GENUINELY INFORMATIVE",
            "Removing NDWI HURTS classification performance.",
            "Despite its 21% importance, NDWI is capturing real spectral",
            "class differences beyond the seasonal moisture shift.",
            "ACTION: Keep NDWI in the feature set. The seasonal artefact",
            "        concern is not supported by the ablation evidence.",
        ]
    else:
        lines += [
            "VERDICT: AMBIGUOUS — results within noise threshold",
            f"Delta F1 = {delta_f1:+.4f} is within ±0.02 threshold.",
            f"Delta Jaccard = {delta_jac:+.4f} is within ±0.05 threshold.",
            "NDWI removal has no statistically meaningful effect on this",
            "dataset. The seasonal artefact concern cannot be confirmed",
            "or ruled out from this experiment alone.",
            "ACTION: Try same-season image pair to separate the moisture",
            "        signal from land-cover information more cleanly.",
        ]

    return "\n".join(lines)


def print_ablation_table_for_thesis(df, ndvi_change_pct=13.52):
    """
    Print the formatted ablation table for the thesis Results chapter.

    Call after run_ablation_study():
        print_ablation_table_for_thesis(ablation_results[0])
    """
    print()
    print("=" * 100)
    print("THESIS TABLE: NDWI Ablation Study Results")
    print(f"Reference NDVI change: {ndvi_change_pct:.2f}%  |  "
          f"4x4 spatial block CV, 50,000 training samples")
    print("=" * 100)
    print(f"{'Experiment':<25} {'N Feat':>6} {'CV F1':>8} {'±Std':>7} "
          f"{'ΔF1':>8} {'Change%':>9} {'ΔChange':>9} {'Jaccard':>9} "
          f"{'ΔJaccard':>10}  {'Verdict':<25}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{row['Experiment']:<25} {row['N Features']:>6} "
              f"{row['Spatial CV F1']:>8} {row['CV Std']:>7} "
              f"{row['Delta F1']:>8} {row['RF Change %']:>9} "
              f"{row['Delta Change']:>9} {row['NDVI-RF Jaccard']:>9} "
              f"{row['Delta Jaccard']:>10}  {row['Verdict']:<25}")
    print("=" * 100)
    print()
    print("Column guide:")
    print("  CV F1      = spatial block (4x4) cross-validation macro-F1  [CITE THIS]")
    print("  Delta F1   = change vs baseline (+ = better, - = worse)")
    print("  Jaccard    = pixel-level agreement between RF and NDVI change masks")
    print("               (higher = RF detects change at same locations as NDVI)")
    print("  Delta Jaccard = change vs baseline (+ = more spatially aligned with NDVI)")
    print("  Verdict    = BETTER if ΔF1>+0.02 OR ΔJaccard>+0.05; WORSE if ΔF1<-0.02")
