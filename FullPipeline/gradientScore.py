import numpy as np
import cv2

def gradient_score(
    image: np.ndarray,
    percentile: float = 99.0,
    ksize: int = 3,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Compute the per-pixel gradient score G(x, y) ∈ [0, 1].
 
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale uint8 or float32, or BGR/RGB colour).
        Colour images are converted to grayscale automatically.
    percentile : float
        Normalisation percentile (default 99).  Values at or above this
        gradient magnitude are clipped to 1.0.  Lowering this makes the
        score more aggressive (more pixels near 1); raising it is more
        conservative.
    ksize : int
        Sobel kernel size (must be 1, 3, 5, or 7).  Larger kernels give
        smoother gradient estimates and are less sensitive to noise.
    blur_ksize : int
        Size of the Gaussian pre-blur kernel applied before gradient
        computation to suppress high-frequency noise.  Must be odd.
        Set to 0 or 1 to skip pre-blurring.
 
    Returns
    -------
    G : np.ndarray  shape (H, W),  dtype float32,  values ∈ [0, 1]
        Per-pixel gradient score map.
    """
    # ── 1. ensure grayscale float32 ───────────────────────────────────────────
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
 
    gray = gray.astype(np.float32)
 
    # ── 2. optional Gaussian pre-blur (noise suppression) ────────────────────
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1   # must be odd
        gray = cv2.GaussianBlur(gray, (k, k), sigmaX=0)
 
    # ── 3. Sobel gradients in x and y ────────────────────────────────────────
    #    cv2.Sobel returns float32 when ddepth=cv2.CV_32F
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
 
    # ── 4. gradient magnitude g(x, y) = sqrt(Ix² + Iy²) ─────────────────────
    g = np.sqrt(Ix ** 2 + Iy ** 2)
 
    # ── 5. normalise by the p-th percentile ──────────────────────────────────
    #    Using np.percentile on the full map; ignores zeros so that flat
    #    regions don't drag the percentile down artificially.
    nonzero = g[g > 0]
    if nonzero.size == 0:                          # degenerate: all-flat image
        return np.zeros_like(g)
    g_p = float(np.percentile(nonzero, percentile))
    if g_p == 0:
        return np.zeros_like(g)
    G = np.clip(g / g_p, 0.0, 1.0).astype(np.float32)
    return G