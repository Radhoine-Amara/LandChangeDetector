"""
random_forest.py
----------------
Method 4: Post-Classification Comparison using Random Forest

Unlike the previous 3 methods which compare raw pixel values directly,
this method:
    1. Classifies EACH image separately into land cover classes
    2. Compares the two classification maps
    3. Produces a transition matrix (what became what)

Pipeline:
    Image_Before → [Random Forest] → Map_Before (Forest, Urban, Bare, Water)
    Image_After  → [Random Forest] → Map_After  (Forest, Urban, Bare, Water)
    Transition   = Compare(Map_Before, Map_After)

The key advantage: results are LABELED and INTERPRETABLE
    "15% of Vegetation became Bare Soil"
    "3% of Bare Soil became Urban"

Since we have no ground truth labels, we use:
    → Unsupervised clustering (KMeans) to auto-generate training samples
    → Then Random Forest learns to classify consistently across both images
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ── Land cover class definitions ──────────────────────────────────
CLASS_LABELS = {
    0: 'Unclassified',
    1: 'Vegetation',
    2: 'Bare Soil / Urban',
    3: 'Water / Shadow',
    4: 'Sparse Vegetation',
}

CLASS_COLORS = {
    0: '#ffffff',   # white
    1: '#1a9850',   # green
    2: '#d73027',   # red-brown
    3: '#2166ac',   # blue
    4: '#fee08b',   # yellow
}


def build_feature_stack(bands, band_names=None):
    """
    Build a feature matrix from a list of bands.
    Adds derived features: NDVI, NDWI for richer classification.

    Parameters
    ----------
    bands      : list of np.ndarray  [B02, B03, B04, B08]
                 Each shape (rows, cols), may contain NaN
    band_names : list of str

    Returns
    -------
    features   : np.ndarray  shape (rows*cols, n_features)
    feature_names : list of str
    valid_mask : np.ndarray bool  shape (rows, cols)
    """

    blue, green, red, nir = bands[0], bands[1], bands[2], bands[3]

    # ── Derived indices ───────────────────────────────────────────
    # NDVI: vegetation index
    denom_ndvi = nir + red
    ndvi = np.where(denom_ndvi == 0, np.nan,
                    (nir - red) / denom_ndvi).astype(np.float32)

    # NDWI: water index (Green - NIR) / (Green + NIR)
    denom_ndwi = green + nir
    ndwi = np.where(denom_ndwi == 0, np.nan,
                    (green - nir) / denom_ndwi).astype(np.float32)

    # NDBI: built-up index — needs SWIR, approximate with Red/NIR ratio
    # Simple brightness as proxy
    brightness = (blue + green + red + nir) / 4.0

    feature_names = ['B02_Blue', 'B03_Green', 'B04_Red', 'B08_NIR',
                     'NDVI', 'NDWI', 'Brightness']

    # ── Stack all features ────────────────────────────────────────
    feature_stack = np.stack([blue, green, red, nir,
                               ndvi, ndwi, brightness], axis=-1)
    # Shape: (rows, cols, n_features)

    # ── Valid pixel mask (no NaN in any feature) ──────────────────
    valid_mask = ~np.any(np.isnan(feature_stack), axis=-1)

    # ── Reshape to 2D: (n_pixels, n_features) ────────────────────
    rows, cols, n_feat = feature_stack.shape
    features_2d = feature_stack.reshape(-1, n_feat)

    return features_2d, feature_names, valid_mask


def generate_training_samples(features_2d, valid_mask,
                               n_classes=4, n_samples=5000,
                               random_state=42):
    """
    Use KMeans clustering to automatically generate training labels.

    Since we have no manual ground truth, KMeans groups pixels
    into spectral clusters, which we then use as training labels
    for the Random Forest.

    Parameters
    ----------
    features_2d  : np.ndarray  shape (n_pixels, n_features)
    valid_mask   : np.ndarray  bool  shape (rows, cols)
    n_classes    : int   number of land cover classes
    n_samples    : int   training samples to generate
    random_state : int

    Returns
    -------
    X_train : np.ndarray  training features
    y_train : np.ndarray  training labels (1-indexed)
    kmeans  : fitted KMeans model
    scaler  : fitted StandardScaler
    """

    # Get valid pixels only
    valid_flat = valid_mask.flatten()
    valid_features = features_2d[valid_flat]

    print(f"Valid pixels for training: {valid_features.shape[0]:,}")

    # Normalize features (important for KMeans)
    scaler = StandardScaler()
    valid_scaled = scaler.fit_transform(valid_features)

    # Sample a subset for clustering (faster)
    n_cluster_samples = min(50000, valid_features.shape[0])
    idx = np.random.RandomState(random_state).choice(
        valid_features.shape[0], n_cluster_samples, replace=False
    )
    sample_scaled = valid_scaled[idx]

    print(f"Running KMeans with {n_classes} clusters "
          f"on {n_cluster_samples:,} samples...")

    kmeans = KMeans(n_clusters=n_classes, random_state=random_state,
                    n_init=10, max_iter=300)
    kmeans.fit(sample_scaled)

    # Predict cluster labels for ALL valid pixels
    print("Predicting cluster labels for all pixels...")
    all_labels = kmeans.predict(valid_scaled)
    all_labels += 1   # shift to 1-indexed (0 = unclassified)

    # Sample n_samples for RF training
    n_train = min(n_samples, len(all_labels))
    train_idx = np.random.RandomState(random_state).choice(
        len(all_labels), n_train, replace=False
    )

    X_train = valid_features[train_idx]
    y_train = all_labels[train_idx]

    print(f"Training samples: {X_train.shape[0]:,}")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt:,} samples")

    return X_train, y_train, kmeans, scaler


def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray  training features
    y_train : np.ndarray  training labels

    Returns
    -------
    clf         : trained RandomForestClassifier
    feature_imp : feature importance array
    """

    print("\nTraining Random Forest...")

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state,
        stratify=y_train
    )

    clf = RandomForestClassifier(
        n_estimators=100,      # 100 trees
        max_depth=15,          # prevent overfitting
        min_samples_leaf=5,
        n_jobs=-1,             # use all CPU cores
        random_state=random_state,
        class_weight='balanced'
    )

    clf.fit(X_tr, y_tr)

    # Validation accuracy
    val_pred = clf.predict(X_val)
    accuracy = (val_pred == y_val).mean()
    print(f"Validation accuracy: {accuracy * 100:.1f}%")

    return clf, accuracy


def classify_image(clf, features_2d, valid_mask, image_shape):
    """
    Apply trained classifier to all valid pixels of an image.

    Parameters
    ----------
    clf          : trained RandomForestClassifier
    features_2d  : np.ndarray  shape (n_pixels, n_features)
    valid_mask   : np.ndarray  bool  shape (rows, cols)
    image_shape  : tuple  (rows, cols)

    Returns
    -------
    class_map : np.ndarray uint8  shape (rows, cols)
        0 = unclassified/masked
        1-4 = land cover class
    """

    rows, cols = image_shape
    class_map = np.zeros(rows * cols, dtype=np.uint8)

    valid_flat = valid_mask.flatten()
    valid_features = features_2d[valid_flat]

    # Predict in batches to avoid memory issues
    batch_size = 100000
    predictions = []

    for i in range(0, len(valid_features), batch_size):
        batch = valid_features[i:i + batch_size]
        pred  = clf.predict(batch)
        predictions.append(pred)
        pct = min(100, (i + batch_size) / len(valid_features) * 100)
        print(f"  Classifying... {pct:.0f}%", end='\r')

    print()
    class_map[valid_flat] = np.concatenate(predictions)
    class_map = class_map.reshape(rows, cols)

    return class_map


def compute_transition_matrix(map_before, map_after, n_classes=4):
    """
    Compute the transition matrix between two classification maps.

    Entry [i, j] = number of pixels that were class i → became class j

    Parameters
    ----------
    map_before : np.ndarray uint8
    map_after  : np.ndarray uint8
    n_classes  : int

    Returns
    -------
    matrix      : np.ndarray  shape (n_classes, n_classes)  raw counts
    matrix_pct  : np.ndarray  shape (n_classes, n_classes)  row percentages
    """

    classes = list(range(1, n_classes + 1))
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)

    valid = (map_before > 0) & (map_after > 0)

    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            matrix[i, j] = ((map_before == ci) &
                             (map_after  == cj) & valid).sum()

    # Row-normalize to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid division by zero
    matrix_pct = (matrix / row_sums) * 100

    return matrix, matrix_pct


def run_random_forest(bands_before, bands_after,
                      n_classes=4, n_train_samples=5000,
                      random_state=42):
    """
    Full Random Forest post-classification comparison pipeline.

    Parameters
    ----------
    bands_before     : list of np.ndarray [B02, B03, B04, B08]
    bands_after      : list of np.ndarray [B02, B03, B04, B08]
    n_classes        : int   number of land cover classes
    n_train_samples  : int   training samples for RF
    random_state     : int

    Returns
    -------
    results : dict with keys:
        'map_before'      → classified map at time 1
        'map_after'       → classified map at time 2
        'change_map'      → pixels where class changed
        'transition_matrix'     → raw counts
        'transition_matrix_pct' → row-normalized percentages
        'clf'             → trained classifier
        'change_pct'      → % of pixels that changed class
    """

    rows, cols = bands_before[0].shape
    image_shape = (rows, cols)

    # ── Step 1: Build feature stacks ─────────────────────────────
    print("=" * 50)
    print("STEP 1: Building feature stacks")
    features_b, feat_names, valid_b = build_feature_stack(bands_before)
    features_a, _,          valid_a = build_feature_stack(bands_after)

    # Combined valid mask
    valid_both_rf = valid_b & valid_a
    print(f"Valid pixels (both dates): {valid_both_rf.sum():,}")

    # ── Step 2: Generate training samples from BEFORE image ──────
    print("\nSTEP 2: Generating training samples via KMeans")
    X_train, y_train, kmeans, scaler = generate_training_samples(
        features_b, valid_both_rf,
        n_classes=n_classes,
        n_samples=n_train_samples,
        random_state=random_state
    )

    # ── Step 3: Train Random Forest ───────────────────────────────
    print("\nSTEP 3: Training Random Forest classifier")
    clf, accuracy = train_random_forest(X_train, y_train, random_state)

    # ── Step 4: Classify both images ─────────────────────────────
    print("\nSTEP 4: Classifying BEFORE image")
    map_before = classify_image(clf, features_b, valid_both_rf, image_shape)

    print("STEP 5: Classifying AFTER image")
    map_after  = classify_image(clf, features_a, valid_both_rf, image_shape)

    # ── Step 5: Change map ────────────────────────────────────────
    valid_pixels = valid_both_rf
    change_map   = np.where(
        valid_pixels,
        (map_before != map_after).astype(np.uint8),
        0
    )
    change_pct = (change_map[valid_pixels].sum() /
                  valid_pixels.sum()) * 100

    # ── Step 6: Transition matrix ─────────────────────────────────
    print("\nSTEP 6: Computing transition matrix")
    matrix, matrix_pct = compute_transition_matrix(
        map_before, map_after, n_classes
    )

    print(f"\nTotal pixels that changed class: "
          f"{change_map.sum():,} ({change_pct:.1f}%)")

    return {
        'map_before'            : map_before,
        'map_after'             : map_after,
        'change_map'            : change_map,
        'transition_matrix'     : matrix,
        'transition_matrix_pct' : matrix_pct,
        'clf'                   : clf,
        'accuracy'              : accuracy,
        'change_pct'            : change_pct,
        'valid_mask'            : valid_both_rf,
        'feature_names'         : feat_names,
        'class_names'           : [CLASS_LABELS[i] for i in range(1, n_classes + 1)],
    }