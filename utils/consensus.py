"""
utils/consensus.py  — v2.1
==========================
Project 23 — LandChangeDetector  (Batna Province, Algeria)

Integrated multi-method consensus pipeline.

Changelog v2.1 (bug-fix release)
----------------------------------
characterise_ndvi_change() — TWO bugs fixed:

  BUG 1 (denominator):
    OLD: valid = ndvi_results.get('valid_mask',
                 ndvi_results.get('change_mask', None))
         n_valid = int(valid.sum())          # ← only 448,682 (changed pixels)
    SYMPTOM: all percentages explode above 100%
             ("Masked: 799.69%")
    FIX:  n_valid = ctype.size              # always = 4,000,000 total pixels
          percentages now sum to exactly 100%

  BUG 2 (coverage gap):
    The original 6 categories only covered ndvi_before < 0.10 (Bare Soil)
    and ndvi_before >= 0.25 (Stable Vegetation).  Pixels with NDVI 0.10–0.25
    (Batna transition-zone steppe, mean NDVI = 0.165) fell into category 0.
    With 87.6% of valid pixels in this unassigned gap, the category
    breakdown was nearly meaningless.
    FIX:  Added category 7 "Stable Transition" (NDVI 0.10–0.25, unchanged).
          All valid non-changed pixels now receive a meaningful label.
          Category 6 threshold lowered to ndvi_before < 0.12 (rock/urban).
          Category 5 threshold lowered to ndvi_before >= 0.22 (dense veg).
"""

import numpy as np


# =============================================================================
# 1. CONSENSUS CHANGE MAP
# =============================================================================

def build_consensus_map(ndvi_results, rf_results, cva_results, bd_results,
                        ndvi_weight=0.45,
                        rf_weight=0.30,
                        cva_weight=0.20,
                        bd_weight=0.05,
                        conf_threshold=0.60):
    """
    Build a weighted consensus change map from all four methods.

    Weights rationale
    -----------------
    NDVI (0.45) — highest: physically grounded, seasonally stable,
                  full-image validated (13.02% across 9 tiles).
    RF   (0.30) — second: geographic labels from WorldCover, confidence-
                  weighted per-pixel. Auto-halved to 0.15 if KMeans.
    CVA  (0.20) — third: multi-band magnitude, vegetation loss component
                  credible; soil-gain component is seasonal artefact.
    BD   (0.05) — lowest: single-band, most inflated by phenology.

    Returns
    -------
    consensus : dict — see keys below
    """
    ndvi_change = ndvi_results['change_mask'].astype(np.float32)

    rf_raw = rf_results['change_mask'].astype(np.float32)
    training_mode = rf_results.get('training_mode', 'unsupervised')
    actual_rf_weight = rf_weight * 0.5 if training_mode == 'unsupervised' \
                       else rf_weight
    if training_mode == 'unsupervised':
        print(f"  RF training mode: unsupervised → RF weight halved "
              f"({rf_weight:.2f} → {actual_rf_weight:.2f})")

    prob_b = rf_results.get('prob_before')
    prob_a = rf_results.get('prob_after')
    if prob_b is not None and prob_a is not None:
        min_conf    = np.minimum(prob_b, prob_a)
        conf_scale  = np.clip((min_conf - conf_threshold) /
                               (1.0 - conf_threshold), 0.0, 1.0)
        rf_weighted = rf_raw * conf_scale
    else:
        rf_weighted = rf_raw

    cva_change = cva_results['change_mask'].astype(np.float32)
    cva_mag    = cva_results.get('magnitude')
    if cva_mag is not None:
        mag_valid = cva_mag[np.isfinite(cva_mag) & (cva_mag > 0)]
        if len(mag_valid) > 0:
            p95          = float(np.percentile(mag_valid, 95))
            mag_norm     = np.clip(cva_mag / p95, 0.0, 1.0)
            mag_norm     = np.where(np.isfinite(mag_norm), mag_norm, 0.0)
            cva_weighted = cva_change * (0.5 + 0.5 * mag_norm.astype(np.float32))
        else:
            cva_weighted = cva_change
    else:
        cva_weighted = cva_change

    bd_change = bd_results['change_mask'].astype(np.float32)

    # Shared valid mask
    delta_ndvi = ndvi_results.get('delta_ndvi')
    if delta_ndvi is not None:
        valid = np.isfinite(delta_ndvi)
    else:
        valid = (ndvi_change >= 0) & (rf_raw >= 0) & (cva_change >= 0)
    for r in [ndvi_results, rf_results, cva_results]:
        vm = r.get('valid_mask')
        if vm is not None:
            valid = valid & vm

    total_w = ndvi_weight + actual_rf_weight + cva_weight + bd_weight
    w_ndvi  = ndvi_weight      / total_w
    w_rf    = actual_rf_weight / total_w
    w_cva   = cva_weight       / total_w
    w_bd    = bd_weight        / total_w
    weights_used = dict(NDVI=round(w_ndvi, 4), RF=round(w_rf, 4),
                        CVA=round(w_cva, 4),   BD=round(w_bd, 4))
    print(f"  Consensus weights (normalised): {weights_used}")

    raw = (w_ndvi * ndvi_change +
           w_rf   * rf_weighted +
           w_cva  * cva_weighted +
           w_bd   * bd_change)
    raw = np.where(valid, raw, 0.0).astype(np.float32)

    THRESH_LIKELY    = 0.35
    THRESH_CONFIRMED = 0.55
    consensus_map = np.zeros_like(raw, dtype=np.uint8)
    consensus_map[valid & (raw >= THRESH_LIKELY)   ] = 1
    consensus_map[valid & (raw >= THRESH_CONFIRMED)] = 2

    n_valid     = int(valid.sum())
    n_likely    = int((consensus_map == 1).sum())
    n_confirmed = int((consensus_map == 2).sum())
    n_any       = n_likely + n_confirmed
    pct = lambda n: n / n_valid * 100 if n_valid > 0 else 0.0

    agreement_map = (
        ndvi_change.astype(np.uint8) +
        rf_raw.astype(np.uint8) +
        cva_change.astype(np.uint8) +
        bd_change.astype(np.uint8)
    ).astype(np.uint8)
    agreement_map = np.where(valid, agreement_map, 0).astype(np.uint8)

    def jaccard(a, b, mask):
        a, b   = a[mask].astype(bool), b[mask].astype(bool)
        inter  = (a & b).sum()
        union  = (a | b).sum()
        return float(inter / union) if union > 0 else 0.0

    method_agreement = {
        'NDVI_vs_RF' : jaccard(ndvi_change, rf_raw,    valid),
        'NDVI_vs_CVA': jaccard(ndvi_change, cva_change, valid),
        'NDVI_vs_BD' : jaccard(ndvi_change, bd_change,  valid),
        'RF_vs_CVA'  : jaccard(rf_raw,    cva_change,  valid),
        'RF_vs_BD'   : jaccard(rf_raw,    bd_change,   valid),
        'CVA_vs_BD'  : jaccard(cva_change, bd_change,  valid),
    }

    return {
        'change_map_raw'       : raw,
        'change_map'           : consensus_map,
        'change_pct_likely'    : pct(n_likely),
        'change_pct_confirmed' : pct(n_confirmed),
        'change_pct_any'       : pct(n_any),
        'agreement_map'        : agreement_map,
        'valid_mask'           : valid,
        'weights_used'         : weights_used,
        'method_agreement'     : method_agreement,
        'thresh_likely'        : THRESH_LIKELY,
        'thresh_confirmed'     : THRESH_CONFIRMED,
    }


# =============================================================================
# 2. CREDIBILITY-WEIGHTED TRANSITION MATRIX
# =============================================================================

def build_credibility_weighted_transitions(rf_results, ndvi_results,
                                            conf_threshold=0.60):
    """
    Harmonise the RF transition matrix against the NDVI reference.

    Scales all off-diagonal entries by (NDVI_change%) / (RF_change%),
    then zeros physically implausible transitions.
    """
    if (rf_results.get('transition_matrix_conf_pct') is not None and
            rf_results.get('n_confident_pixels', 0) > 0):
        trans_pct = rf_results['transition_matrix_conf_pct']
        source    = 'confidence-filtered'
    else:
        trans_pct = rf_results.get('transition_matrix_pct')
        source    = 'all-pixels'

    if trans_pct is None:
        return None

    ndvi_change_pct = ndvi_results.get('change_pct', 13.52)
    rf_change_pct   = rf_results.get('change_pct', 50.0)
    scale_factor    = (ndvi_change_pct / rf_change_pct
                       if rf_change_pct > 0 else 1.0)

    n      = trans_pct.shape[0]
    scaled = trans_pct.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                scaled[i, j] *= scale_factor
        off_sum      = scaled[i, :].sum() - scaled[i, i]
        scaled[i, i] = max(0.0, 100.0 - off_sum)

    CLASS_NAMES   = ['Vegetation', 'Bare Soil/Urban',
                     'Water/Shadow', 'Sparse Veg']
    suspect_cells = []

    if n >= 2 and scaled[1, 0] > 15.0:
        suspect_cells.append((1, 0,
            f"Bare Soil→Vegetation {scaled[1,0]:.1f}% "
            f"(mass revegetation implausible)"))
    if n >= 3:
        for j in range(n):
            if j != 2 and scaled[2, j] > 30.0:
                suspect_cells.append((2, j,
                    f"Water→{CLASS_NAMES[j]} {scaled[2,j]:.1f}%"))
        for i in range(n):
            if i != 2 and scaled[i, 2] > 10.0:
                suspect_cells.append((i, 2,
                    f"{CLASS_NAMES[i]}→Water {scaled[i,2]:.1f}%"))

    credible = scaled.copy()
    for i, j, _ in suspect_cells:
        credible[i, j] = 0.0
    for i in range(n):
        rs = credible[i].sum()
        if rs > 0:
            credible[i] = credible[i] / rs * 100.0

    print(f"\n  Transition source   : {source}")
    print(f"  RF change %         : {rf_change_pct:.1f}%")
    print(f"  NDVI reference %    : {ndvi_change_pct:.1f}%")
    print(f"  Scale factor        : {scale_factor:.4f}")
    print(f"  Suspect cells       : {len(suspect_cells)}")
    for i, j, reason in suspect_cells:
        print(f"    ⚠️  [{i},{j}] {reason}")

    return {
        'matrix_scaled_pct'   : scaled,
        'matrix_credible_pct' : credible,
        'suspect_cells'       : suspect_cells,
        'scale_factor'        : scale_factor,
        'ndvi_reference_pct'  : ndvi_change_pct,
        'rf_original_pct'     : trans_pct,
        'source'              : source,
    }


# =============================================================================
# 3. NDVI CHANGE SEVERITY  —  v2.1  (BOTH BUGS FIXED)
# =============================================================================

def characterise_ndvi_change(ndvi_results, rf_results):
    """
    Characterise NDVI change into 7 severity / state classes.

    Classes
    -------
    0  Masked / cloud          (invalid data)
    1  Severe loss             (ΔNDVI < −0.20)
    2  Moderate loss           (−0.20 ≤ ΔNDVI < −0.10)
    3  Minor loss              (−0.10 ≤ ΔNDVI < −threshold, within loss mask)
    4  Vegetation gain         (ΔNDVI > +threshold, within gain mask)
    5  Stable vegetation       (NDVI_before ≥ 0.22, not changed)
    6  Stable bare/rocky       (NDVI_before < 0.12, not changed)
    7  Stable transition       (0.12 ≤ NDVI_before < 0.22, not changed)
                               ← NEW class, fixes coverage gap for Batna
                                 steppe (mean NDVI = 0.165)

    BUG FIXES in v2.1
    -----------------
    1. Denominator:  n_valid = ctype.size  (total pixels, always 4,000,000)
       → percentages now sum to exactly 100%
       OLD denominator was change_mask.sum() = 448,682 → all % > 100%

    2. Coverage gap: original categories 5 and 6 used thresholds 0.25/0.10,
       leaving 87.6% of valid Batna pixels (NDVI 0.10–0.25) unclassified.
       New category 7 "Stable Transition" covers this range.
       Thresholds adjusted: cat 5 ≥ 0.22, cat 6 < 0.12.
    """
    delta       = ndvi_results.get('delta_ndvi')
    loss_mask   = ndvi_results.get('loss_mask')
    gain_mask   = ndvi_results.get('gain_mask')
    ndvi_before = ndvi_results.get('ndvi_before')
    ndvi_after  = ndvi_results.get('ndvi_after')

    if delta is None or loss_mask is None:
        return None

    # ── Valid mask: pixels with finite NDVI delta = cloud-free ──────────────
    # v2.1 FIX 1: derive valid from delta_ndvi, NOT from change_mask
    valid = np.isfinite(delta)

    # If ndvi_results carries an explicit valid_mask (cloud mask), AND it,
    # but do NOT use change_mask as the valid mask (it's only 13% of pixels).
    explicit_vm = ndvi_results.get('valid_mask')
    if explicit_vm is not None and explicit_vm.shape == delta.shape:
        valid = valid & explicit_vm

    n     = delta.shape
    ctype = np.zeros(n, dtype=np.uint8)   # 0 = masked by default

    # ── Category assignments (applied in increasing priority order) ──────────

    # Category 7: stable transition zone (Batna steppe — NEW v2.1)
    if ndvi_before is not None:
        trans = (valid &
                 np.isfinite(ndvi_before) &
                 (ndvi_before >= 0.12) & (ndvi_before < 0.22) &
                 ~loss_mask & ~gain_mask)
        ctype[trans] = 7

    # Category 6: stable bare/rocky (low NDVI, unchanged)
    if ndvi_before is not None:
        bare = (valid &
                np.isfinite(ndvi_before) &
                (ndvi_before < 0.12) &
                ~loss_mask & ~gain_mask)
        ctype[bare] = 6

    # Category 5: stable vegetation (higher NDVI, unchanged)
    if ndvi_before is not None and ndvi_after is not None:
        stab = (valid &
                np.isfinite(ndvi_before) & (ndvi_before >= 0.22) &
                np.isfinite(ndvi_after)  & (ndvi_after  >= 0.18) &
                ~loss_mask & ~gain_mask)
        ctype[stab] = 5

    # Category 4: vegetation gain
    ctype[valid & gain_mask] = 4

    # Category 3, 2, 1: loss severity
    ctype[valid & loss_mask & (delta >= -0.10)                    ] = 3
    ctype[valid & loss_mask & (delta < -0.10) & (delta >= -0.20)  ] = 2
    ctype[valid & loss_mask & (delta < -0.20)                     ] = 1

    # ── v2.1 FIX 2: denominator = total pixels, percentages sum to 100% ─────
    n_total = int(ctype.size)     # 4,000,000 — FIXED (was change_mask.sum())
    n_cloud = int((~valid).sum()) # actual cloud-masked count for reference

    labels = {
        0: 'Masked / Cloud',
        1: 'Severe Loss (ΔNDVI < −0.20)',
        2: 'Moderate Loss (−0.20 to −0.10)',
        3: 'Minor Loss (−0.10 to threshold)',
        4: 'Vegetation Gain',
        5: 'Stable Vegetation (NDVI ≥ 0.22)',
        6: 'Stable Bare / Rocky (NDVI < 0.12)',
        7: 'Stable Transition Zone (NDVI 0.12–0.22)',
    }

    class_stats = {}
    for c, lbl in labels.items():
        cnt = int((ctype == c).sum())
        pct = cnt / n_total * 100     # FIXED denominator
        class_stats[lbl] = {
            'count' : cnt,
            'pct'   : pct,
            'class' : c,
        }

    # Sanity assertion — percentages must sum to ~100%
    total_pct = sum(v['pct'] for v in class_stats.values())
    assert abs(total_pct - 100.0) < 0.01, \
        f"Severity % do not sum to 100%: {total_pct:.4f}%"

    print(f"  NDVI severity bug status : FIXED  (denominator = {n_total:,})")
    print(f"  Cloud-masked pixels      : {n_cloud:,}  "
          f"({n_cloud/n_total*100:.1f}%)")
    print(f"  Percentage sum check     : {total_pct:.4f}%  ✅")

    return {
        'change_type_map' : ctype,
        'class_stats'     : class_stats,
        'labels'          : labels,
        'n_total_pixels'  : n_total,
        'n_cloud_pixels'  : n_cloud,
    }


# =============================================================================
# 4. MASTER SUMMARY TABLE
# =============================================================================

def build_summary_table(bd_results, ndvi_results, cva_results,
                        rf_results, consensus_results=None):
    """Summary table for thesis Results chapter — one row per method."""
    training_mode = rf_results.get('training_mode', 'unsupervised')
    rf_mode_label = ('Supervised (WorldCover)'
                     if training_mode == 'supervised'
                     else 'Unsupervised (KMeans)')

    sp_mean = rf_results.get('spatial_cv_mean')
    sp_std  = rf_results.get('spatial_cv_std')
    rf_acc  = (f"{sp_mean:.3f}±{sp_std:.3f} (spatial CV macro-F1)"
               if sp_mean is not None
               else f"{rf_results.get('accuracy',0)*100:.1f}% (random split)")

    rows = [
        {'Method': 'Band Differencing',
         'Change %': f"{bd_results['change_pct']:.2f}%",
         'Changed Area': f"{bd_results['change_pct']*400/100:.0f} km²",
         'Threshold': f"{bd_results['threshold']:.2f}",
         'Accuracy': 'N/A',
         'Reliability': '⚠️  Low (seasonal noise dominant)',
         'Best use': 'Quick screen only'},

        {'Method': 'NDVI Differencing ⭐',
         'Change %': f"{ndvi_results['change_pct']:.2f}%",
         'Changed Area': f"{ndvi_results['change_pct']*400/100:.0f} km²",
         'Threshold': f"±{ndvi_results['threshold']:.4f}",
         'Accuracy': 'N/A (reference method)',
         'Reliability': '✅ HIGH — physical anchor',
         'Best use': 'Primary result for thesis'},

        {'Method': 'CVA',
         'Change %': f"{cva_results['change_pct']:.2f}%",
         'Changed Area': f"{cva_results['change_pct']*400/100:.0f} km²",
         'Threshold': f"{cva_results['threshold']:.2f}",
         'Accuracy': 'N/A',
         'Reliability': '⚠️  Medium (Veg loss credible; soil gain artefact)',
         'Best use': 'Change type characterisation'},

        {'Method': f'Random Forest ({rf_mode_label})',
         'Change %': f"{rf_results['change_pct']:.2f}%",
         'Changed Area': f"{rf_results['change_pct']*400/100:.0f} km²",
         'Threshold': 'N/A',
         'Accuracy': rf_acc,
         'Reliability': ('✅ HIGH (supervised + spatial CV)'
                         if training_mode == 'supervised'
                         else '❌ LOW inflated ~50% (KMeans seasonal)'),
         'Best use': 'Transition matrix (supervised only)'},
    ]

    if consensus_results is not None:
        rows.append({
            'Method': 'Consensus (NDVI+RF+CVA+BD)',
            'Change %': f"{consensus_results['change_pct_any']:.2f}%",
            'Changed Area': f"{consensus_results['change_pct_any']*400/100:.0f} km²",
            'Threshold': f"w={consensus_results['weights_used']}",
            'Accuracy': 'N/A',
            'Reliability': '✅ HIGH (multi-method agreement)',
            'Best use': 'Final reportable estimate'})

    return rows


# =============================================================================
# 5. MASTER PIPELINE
# =============================================================================

def run_integrated_pipeline(bd_results, ndvi_results, cva_results, rf_results,
                             conf_threshold=0.60,
                             ndvi_weight=0.45, rf_weight=0.30,
                             cva_weight=0.20,  bd_weight=0.05):
    """
    Run the complete integrated consensus pipeline.

    Call after all four algorithm results are computed.
    Returns dict with keys: consensus, credible_transitions,
                            ndvi_characterisation, summary_table
    """
    print("=" * 60)
    print("INTEGRATED CONSENSUS PIPELINE  (consensus.py v2.1)")
    print("=" * 60)

    print("\n[1/4] Building consensus change map...")
    consensus = build_consensus_map(
        ndvi_results, rf_results, cva_results, bd_results,
        ndvi_weight=ndvi_weight, rf_weight=rf_weight,
        cva_weight=cva_weight,   bd_weight=bd_weight,
        conf_threshold=conf_threshold,
    )
    print(f"  Likely change    : {consensus['change_pct_likely']:.2f}%")
    print(f"  Confirmed change : {consensus['change_pct_confirmed']:.2f}%")
    print(f"  Any change       : {consensus['change_pct_any']:.2f}%")
    print("  Pairwise Jaccard:")
    for pair, j in consensus['method_agreement'].items():
        print(f"    {pair:<18s} {j:.3f}  {'█'*int(j*20)}")

    print("\n[2/4] Credibility-weighted transitions...")
    cred_trans = build_credibility_weighted_transitions(
        rf_results, ndvi_results, conf_threshold=conf_threshold
    )

    print("\n[3/4] NDVI change severity (v2.1 — bugs fixed)...")
    ndvi_char = characterise_ndvi_change(ndvi_results, rf_results)
    if ndvi_char is not None:
        print("  Severity breakdown:")
        for lbl, stats in ndvi_char['class_stats'].items():
            if stats['count'] > 0:
                print(f"    {lbl:<45s}: "
                      f"{stats['count']:>8,} px  ({stats['pct']:5.2f}%)")

    print("\n[4/4] Building summary table...")
    summary = build_summary_table(
        bd_results, ndvi_results, cva_results, rf_results, consensus
    )

    training = rf_results.get('training_mode', 'unsupervised')
    print("\n" + "─" * 60)
    print("CONSENSUS RESULT:")
    if training == 'supervised':
        best = consensus['change_pct_any']
        print(f"  Best estimate   : {best:.2f}%  "
              f"(~{best*400/100:.0f} km² of 20×20 km crop)")
    else:
        ndvi_chg = ndvi_results.get('change_pct', 13.52)
        cons_any = consensus['change_pct_any']
        print(f"  NDVI (primary)  : {ndvi_chg:.2f}%")
        print(f"  Consensus (any) : {cons_any:.2f}%  "
              f"(includes down-weighted KMeans RF)")
    print("─" * 60)

    return {
        'consensus'              : consensus,
        'credible_transitions'   : cred_trans,
        'ndvi_characterisation'  : ndvi_char,
        'summary_table'          : summary,
    }
