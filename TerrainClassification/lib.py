"""
Mars Terrain Classification — Python Implementation
Based on: Lv et al. (2022), "Highly Accurate Visual Method of Mars Terrain
Classification for Rovers Based on Novel Image Features", Entropy 24(9), 1304.

Pipeline
--------
1. Load a grayscale Mars terrain image
2. For every pixel, extract a 75-dimensional feature vector using five
   novel feature types at three window scales
3. Train a Random Forest classifier on labelled pixels
4. Predict a terrain label for every pixel → terrain matrix

Terrain labels:
    0 = Sandy Terrain   (ST)  — soft sand, rover sinkage risk
    1 = Hard Terrain    (HT)  — bedrock/slate, safest for rover
    2 = Gravel Terrain  (GT)  — hard gravel, wheel damage risk
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SECTION 1 — Feature Extraction Utilities
# ─────────────────────────────────────────────

def extract_gradient_image(gray: np.ndarray) -> np.ndarray:
    """
    Compute the pixel-wise gradient magnitude of a grayscale image.

    The paper defines (Equations 1 & 2):
        g_u(u,v) = f(u+1, v) - f(u, v)   ← horizontal difference
        g_v(u,v) = f(u, v+1) - f(u, v)   ← vertical difference
        g(u,v)   = sqrt(g_u^2 + g_v^2)   ← Euclidean magnitude

    This is equivalent to the Sobel/finite-difference gradient magnitude
    and is highest for GT (gravel), medium for HT, lowest for ST (sand).
    """
    # np.roll shifts the array by 1 in the given axis, giving f(u+1,v)
    gu = np.roll(gray.astype(np.float32), -1, axis=0) - gray.astype(np.float32)
    gv = np.roll(gray.astype(np.float32), -1, axis=1) - gray.astype(np.float32)
    return np.sqrt(gu**2 + gv**2)


def msgggf(window: np.ndarray, n_levels: int = 10, dg: float = 5.0) -> np.ndarray:
    """
    Multiscale Gray Gradient-Grade Features (MSGGGFs) — Section 3.1.

    Idea: Count what fraction of pixels in the window exceed each gradient
    threshold. Gravel has many high-gradient pixels; sand has almost none.

    Equation 3: threshold_j = j * dg         (j = 1..10, dg = 5)
    Equation 4: p_gj = N_gj / n^2            (pixel proportion at grade j)

    Returns a 10-D feature vector (one value per gradient level).
    """
    grad = extract_gradient_image(window)
    n_pixels = window.size
    features = np.zeros(n_levels)
    for j in range(1, n_levels + 1):
        threshold = j * dg
        features[j - 1] = np.sum(grad > threshold) / n_pixels
    return features


def msesgf(window: np.ndarray, n_levels: int = 9, de: float = 0.1) -> np.ndarray:
    """
    Multiscale Edges Strength-Grade Features (MSESGFs) — Section 3.2.

    Idea: Apply Canny edge detection at 9 different sensitivity thresholds.
    A strict threshold (high value) only detects strong edges; a loose one
    catches weak edges too. The proportion of edge pixels at each level
    encodes edge density and strength.

    Equation 5: threshold_ej = j * de        (j = 1..9, de = 0.1)
    Equation 7: p_ej = N_ej / n^2

    The Canny low/high threshold ratio is kept at 2:1 (standard practice).
    """
    n_pixels = window.size
    # Normalize window to 0–255 uint8 for cv2.Canny
    w_norm = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    features = np.zeros(n_levels)
    for j in range(1, n_levels + 1):
        # Stronger j → higher threshold → only the strongest edges survive
        high = max(j * de * 255, 1.0)
        low = high * 0.5
        edges = cv2.Canny(w_norm, low, high)
        features[j - 1] = np.sum(edges > 0) / n_pixels
    return features


def msfdmaf(window: np.ndarray) -> float:
    """
    Multiscale Frequency-Domain Mean Amplitude Feature (MSFDMAF) — Sec 3.3.2.

    Idea: Fourier-transform the window and take the mean of all amplitude
    values. Sandy terrain has a low-energy, concentrated spectrum (low mean).
    Gravel has energy spread across all frequencies (high mean).

    Equation 8: p_A = sum(A(u,v)) / n^2
    """
    spectrum = fftshift(fft2(window.astype(np.float32)))
    amplitude = np.abs(spectrum)
    return float(np.mean(amplitude))


def msssf(window: np.ndarray) -> np.ndarray:
    """
    Multiscale Spectrum Symmetry Features (MSSSFs) — Section 3.3.3.

    Idea: Split the 2-D frequency spectrum into four quadrants and measure
    how asymmetric the distribution is. Sandy terrain has a nearly symmetric
    spectrum because sand grains scatter light isotropically in all directions.
    Gravel and hard terrain break this symmetry (directional texture).

    Equations 9–11: compare mean and std of quadrant 1 vs quadrants 2 and 4.
    Returns a 4-D vector: [|m1-m2|, |m1-m4|, |σ1-σ2|, |σ1-σ4|]
    """
    spectrum = fftshift(fft2(window.astype(np.float32)))
    amplitude = np.abs(spectrum)
    cy, cx = amplitude.shape[0] // 2, amplitude.shape[1] // 2

    # Four quadrants centred at the DC component
    q1 = amplitude[cy:, cx:]    # bottom-right  (wu≥0, wv≥0)
    q2 = amplitude[cy:, :cx]    # bottom-left   (wu<0, wv≥0)
    q4 = amplitude[:cy, cx:]    # top-right     (wu≥0, wv<0)

    # Safe stats in case a quadrant is empty (very small windows)
    m1, m2, m4 = np.mean(q1), np.mean(q2), np.mean(q4)
    s1 = np.std(q1) if q1.size > 1 else 0.0
    s2 = np.std(q2) if q2.size > 1 else 0.0
    s4 = np.std(q4) if q4.size > 1 else 0.0

    return np.array([
        abs(m1 - m2),   # p_Fx  — mean asymmetry along u-axis
        abs(m1 - m4),   # p_Fy  — mean asymmetry along v-axis
        abs(s1 - s2),   # p_σx  — std asymmetry along u-axis
        abs(s1 - s4),   # p_σy  — std asymmetry along v-axis
    ])


def mssamf(window: np.ndarray) -> float:
    """
    Multiscale Spectrum Amplitude-Moment Feature (MSSAMF) — Section 3.3.4.

    Idea: Multiply each frequency bin's amplitude by its distance from the
    DC centre, then average. This is a "spectral moment" — it measures how
    far from DC the spectral energy sits. Sandy terrain concentrates energy
    near DC (low moment). Gravel spreads energy outward (high moment).

    Equation 12: p_m = sum(A(u,v) * d(u,v)) / n^2
    where d(u,v) = Euclidean distance from (u,v) to the centre.
    """
    spectrum = fftshift(fft2(window.astype(np.float32)))
    amplitude = np.abs(spectrum)
    h, w = amplitude.shape
    cy, cx = h / 2, w / 2

    # Build distance matrix from DC centre
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((ys - cy)**2 + (xs - cx)**2)

    moment = np.sum(amplitude * dist) / (h * w)
    return float(moment)


# ─────────────────────────────────────────────
# SECTION 2 — Per-Pixel Feature Extraction
# ─────────────────────────────────────────────

WINDOW_SCALES = [5, 10, 30]   # Three scales used in the paper


def extract_pixel_features(gray: np.ndarray, row: int, col: int) -> np.ndarray:
    """
    Extract the full 75-dimensional feature vector P for a single pixel.

    Feature structure (75 total):
        MSGGGF: 10 × 3 scales = 30
        MSESGF:  9 × 3 scales = 27
        MSFDMAF: 1 × 3 scales =  3
        MSSSF:   4 × 3 scales = 12  (12 values: 4 symmetry features per scale)
        MSSAMF:  1 × 3 scales =  3
        Total                  = 75

    Each window is extracted by padding the image at its borders so pixels
    near the edge still get a full window (reflect101 padding).
    """
    h, w = gray.shape

    msgggf_feats = []
    msesgf_feats = []
    msfdmaf_feats = []
    msssf_feats = []
    mssamf_feats = []

    for n in WINDOW_SCALES:
        half = n // 2
        # Clamp window bounds and extract patch (handles borders)
        r0, r1 = max(0, row - half), min(h, row + half + 1)
        c0, c1 = max(0, col - half), min(w, col + half + 1)
        patch = gray[r0:r1, c0:c1]

        # If patch is too small (border pixels), pad with reflection
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            patch = np.pad(patch, ((0, max(0, 3 - patch.shape[0])),
                                   (0, max(0, 3 - patch.shape[1]))),
                           mode='reflect')

        msgggf_feats.append(msgggf(patch))
        msesgf_feats.append(msesgf(patch))
        msfdmaf_feats.append(msfdmaf(patch))
        msssf_feats.append(msssf(patch))
        mssamf_feats.append(mssamf(patch))

    return np.concatenate([
        *msgggf_feats,    # 30 values
        *msesgf_feats,    # 27 values
        msfdmaf_feats,    #  3 values
        *msssf_feats,     # 12 values
        mssamf_feats,     #  3 values
    ])


def extract_all_features(gray: np.ndarray,
                          stride: int = 1,
                          verbose: bool = True) -> np.ndarray:
    """
    Extract feature vectors for every pixel (or every `stride`-th pixel).

    Parameters
    ----------
    gray   : H×W uint8 grayscale image
    stride : Process every `stride`-th pixel for faster testing.
             stride=1 → full resolution; stride=4 → 4× faster, lower res.
    verbose: Print progress every 10%

    Returns
    -------
    features : (N, 75) float32 array where N = number of sampled pixels
    coords   : (N, 2) int array of (row, col) for each sample
    """
    h, w = gray.shape
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    N = len(rows) * len(cols)

    features = np.zeros((N, 75), dtype=np.float32)
    coords = np.zeros((N, 2), dtype=np.int32)

    idx = 0
    total_rows = len(rows)
    for i, r in enumerate(rows):
        if verbose and i % max(1, total_rows // 10) == 0:
            print(f"  Feature extraction: {100*i//total_rows}% ({i}/{total_rows} rows)")
        for c in cols:
            features[idx] = extract_pixel_features(gray, r, c)
            coords[idx] = [r, c]
            idx += 1

    if verbose:
        print(f"  Feature extraction: 100% complete. Shape: {features.shape}")
    return features, coords


# ─────────────────────────────────────────────
# SECTION 3 — Classifier Wrappers
# ─────────────────────────────────────────────

TERRAIN_LABELS = {0: "Sandy (ST)", 1: "Hard (HT)", 2: "Gravel (GT)"}
TERRAIN_COLORS = {0: (194, 178, 128),   # sandy tan (BGR)
                  1: (100, 100, 100),   # gray slate
                  2: (60,  100,  60)}   # earthy green


def build_classifier(name: str = "rf"):
    """
    Build one of the three classifiers from the paper.

    Parameters
    ----------
    name : "rf" → Random Forest (best: 94.66%)
           "svm" → Support Vector Machine (89.58%)
           "knn" → K-Nearest Neighbour (89.63%)
    """
    if name == "rf":
        # Paper uses 5 trees; real use → increase n_estimators for stability
        return RandomForestClassifier(n_estimators=100,
                                      max_features="sqrt",
                                      random_state=42,
                                      n_jobs=-1)
    elif name == "svm":
        # RBF kernel for nonlinear separation
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    elif name == "knn":
        # k=5 is a common default; paper doesn't specify K explicitly
        return KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier '{name}'. Use 'rf', 'svm', or 'knn'.")


# ─────────────────────────────────────────────
# SECTION 4 — Full Pipeline
# ─────────────────────────────────────────────

def build_terrain_matrix(image_path: str,
                          labeled_pixels: dict | None = None,
                          classifier_name: str = "rf",
                          stride: int = 1,
                          verbose: bool = True) -> np.ndarray:
    """
    Main function: given an image, return a per-pixel terrain matrix.

    Parameters
    ----------
    image_path      : Path to a grayscale or colour Mars terrain image.
    labeled_pixels  : Optional dict mapping (row, col) → label (0=ST,1=HT,2=GT).
                      If None, a synthetic toy dataset is generated for demo.
    classifier_name : "rf", "svm", or "knn"
    stride          : Pixel stride for feature extraction.
                      stride=1 → full res (slow); stride=8 → fast demo.
    verbose         : Print progress info.

    Returns
    -------
    terrain_matrix : H×W int array (or H//stride × W//stride if stride>1)
                     with values 0=ST, 1=HT, 2=GT
    classifier     : The trained sklearn classifier (for reuse / inspection)
    """
    # ── Step 1: Load image ──────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if verbose:
        print(f"[1/4] Image loaded: {gray.shape} pixels")

    # ── Step 2: Extract features for ALL pixels ─────────────────────────
    if verbose:
        print(f"[2/4] Extracting features (stride={stride})…")
    all_feats, all_coords = extract_all_features(gray, stride=stride, verbose=verbose)

    # ── Step 3: Train classifier on labelled pixels ──────────────────────
    if verbose:
        print(f"[3/4] Training {classifier_name.upper()} classifier…")

    if labeled_pixels is None:
        # Demo mode: label 9 representative corner/centre patches
        # In real use, load your ground-truth labels from annotation files
        if verbose:
            print("      No labels provided — using synthetic demo labels.")
        labeled_pixels = _generate_demo_labels(gray)

    # Build training set from the labeled pixel coordinates
    X_train, y_train = [], []
    for (r, c), label in labeled_pixels.items():
        feat = extract_pixel_features(gray, r, c)
        X_train.append(feat)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    clf = build_classifier(classifier_name)
    clf.fit(X_train, y_train)
    if verbose:
        # StratifiedKFold requires at least n_splits samples in every class.
        # Cap n_splits to the smallest class count to avoid the split error.
        from collections import Counter
        min_class_count = min(Counter(y_train).values())
        n_splits = max(2, min(5, min_class_count))
        cv_scores = cross_val_score(clf, X_train, y_train, cv=n_splits)
        print(f"      Cross-val accuracy (train set, cv={n_splits}): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Step 4: Predict terrain label for every pixel ───────────────────
    if verbose:
        print(f"[4/4] Predicting terrain labels for {len(all_feats):,} pixels…")

    predictions = clf.predict(all_feats)

    # Reshape predictions back into a 2-D matrix
    h, w = gray.shape
    rows_used = np.arange(0, h, stride)
    cols_used = np.arange(0, w, stride)
    terrain_matrix = np.full((len(rows_used), len(cols_used)), -1, dtype=np.int32)

    idx = 0
    for i in range(len(rows_used)):
        for j in range(len(cols_used)):
            terrain_matrix[i, j] = predictions[idx]
            idx += 1

    if verbose:
        print("\n── Terrain Matrix Summary ──────────────────────────────")
        print(f"   Shape : {terrain_matrix.shape}")
        for label, name in TERRAIN_LABELS.items():
            count = np.sum(terrain_matrix == label)
            pct = 100 * count / terrain_matrix.size
            print(f"   {name:<18} : {count:>7,} pixels ({pct:.1f}%)")
        print("────────────────────────────────────────────────────────")

    return terrain_matrix, clf


# ─────────────────────────────────────────────
# SECTION 5 — Visualisation Helpers
# ─────────────────────────────────────────────

def colorize_terrain_matrix(terrain_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a terrain label matrix into a colour image for visualisation.

    Colours (BGR):
        ST (0) → sandy tan    (128, 178, 194)
        HT (1) → slate gray   (100, 100, 100)
        GT (2) → earthy green ( 60, 100,  60)
    """
    h, w = terrain_matrix.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in TERRAIN_COLORS.items():
        mask = terrain_matrix == label
        color_img[mask] = color
    return color_img


def save_terrain_overlay(original_image_path: str,
                          terrain_matrix: np.ndarray,
                          output_path: str,
                          alpha: float = 0.5) -> None:
    """
    Blend the terrain colour map over the original image and save.

    Parameters
    ----------
    original_image_path : Source image path
    terrain_matrix      : H×W label matrix from build_terrain_matrix()
    output_path         : Where to save the blended result
    alpha               : Opacity of the terrain overlay (0=invisible, 1=solid)
    """
    img = cv2.imread(original_image_path)
    color_map = colorize_terrain_matrix(terrain_matrix)

    # Resize overlay to match original if stride was used
    if color_map.shape[:2] != img.shape[:2]:
        color_map = cv2.resize(color_map, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    blended = cv2.addWeighted(img, 1 - alpha, color_map, alpha, 0)

    # Add a simple legend
    legend_items = [(0, "Sandy (ST)"), (1, "Hard (HT)"), (2, "Gravel (GT)")]
    for i, (label, name) in enumerate(legend_items):
        x, y = 10, 20 + i * 22
        color = TERRAIN_COLORS[label]
        cv2.rectangle(blended, (x, y - 10), (x + 16, y + 4), color, -1)
        cv2.putText(blended, name, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, blended)
    print(f"Overlay saved → {output_path}")


# ─────────────────────────────────────────────
# SECTION 6 — Demo / Helper
# ─────────────────────────────────────────────

def _generate_demo_labels(gray: np.ndarray) -> dict:
    """
    Generate synthetic training labels for demo purposes when no annotations
    are available.  Labels ~30 pixels from 3 regions of the image assuming
    the image has some spatial variation (top=sand, middle=hard, bottom=gravel).
    In real use, replace this with your actual ground-truth labels.
    """
    h, w = gray.shape
    labels = {}
    rng = np.random.default_rng(42)

    # Sandy region — sample from top third of image
    for _ in range(10):
        r = rng.integers(0, h // 3)
        c = rng.integers(0, w)
        labels[(int(r), int(c))] = 0   # ST

    # Hard terrain — sample from middle third
    for _ in range(10):
        r = rng.integers(h // 3, 2 * h // 3)
        c = rng.integers(0, w)
        labels[(int(r), int(c))] = 1   # HT

    # Gravel terrain — sample from bottom third
    for _ in range(10):
        r = rng.integers(2 * h // 3, h)
        c = rng.integers(0, w)
        labels[(int(r), int(c))] = 2   # GT

    return labels


def demo_with_synthetic_image() -> np.ndarray:
    """
    Run the full pipeline on a synthetic 128×128 test image.
    The image is divided into three horizontal bands (sand / hard / gravel)
    with realistic gradient and noise characteristics for each terrain type.

    Returns the (16×16) terrain matrix (stride=8 for speed).
    """
    print("=" * 56)
    print("  Mars Terrain Classifier — Demo (synthetic image)")
    print("=" * 56)

    h, w = 128, 128
    img = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(0)

    # Sandy top band: low gradient, uniform, mostly flat
    img[:43, :] = 80 + rng.normal(0, 3, (43, w))

    # Hard terrain middle band: moderate texture with some sharp features
    base = np.zeros((43, w), dtype=np.float32)
    for _ in range(30):
        rr = rng.integers(0, 43)
        cc = rng.integers(0, w)
        base[rr:rr+4, cc:cc+4] = rng.uniform(30, 80)
    img[43:86, :] = 120 + base + rng.normal(0, 5, (43, w))

    # Gravel bottom band: high gradient, high frequency noise
    img[86:, :] = 100 + rng.normal(0, 25, (42, w))

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Save synthetic image temporarily
    tmp_path = "/tmp/synthetic_mars.png"
    cv2.imwrite(tmp_path, img)

    # Ground-truth labels (we know the bands)
    labeled = {}
    for _ in range(15):
        r = int(rng.integers(2, 40))
        c = int(rng.integers(2, w - 2))
        labeled[(r, c)] = 0  # sandy top

    for _ in range(15):
        r = int(rng.integers(45, 83))
        c = int(rng.integers(2, w - 2))
        labeled[(r, c)] = 1  # hard middle

    for _ in range(15):
        r = int(rng.integers(88, 126))
        c = int(rng.integers(2, w - 2))
        labeled[(r, c)] = 2  # gravel bottom

    terrain_matrix, clf = build_terrain_matrix(
        image_path=tmp_path,
        labeled_pixels=labeled,
        classifier_name="rf",
        stride=8,
        verbose=True,
    )

    print("\nTerrain Matrix (0=Sandy, 1=Hard, 2=Gravel):")
    print(terrain_matrix)

    return terrain_matrix


# ─────────────────────────────────────────────
# SECTION 7 — Usage Example
# ─────────────────────────────────────────────
"""
── Real image usage ──────────────────────────────────────────────────────────

from mars_terrain_classifier import build_terrain_matrix, save_terrain_overlay

# 1. Define your ground-truth labels (row, col) → 0/1/2
#    These come from human annotation of a small subset of pixels.
labeled_pixels = {
    (10, 20): 0,   # sandy
    (10, 50): 0,
    (80, 30): 1,   # hard terrain
    (80, 90): 1,
    (200, 40): 2,  # gravel
    (200, 80): 2,
    # ... more labels
}

# 2. Run the classifier
terrain_matrix, clf = build_terrain_matrix(
    image_path="my_mars_image.png",
    labeled_pixels=labeled_pixels,
    classifier_name="rf",   # or "svm" or "knn"
    stride=1,               # 1 = full resolution
)

# 3. terrain_matrix is an H×W numpy array with values 0, 1, or 2

# 4. Optional: save a colour overlay
save_terrain_overlay("my_mars_image.png", terrain_matrix, "output_overlay.png")

─────────────────────────────────────────────────────────────────────────────
"""

if __name__ == "__main__":
    terrain_matrix = demo_with_synthetic_image()