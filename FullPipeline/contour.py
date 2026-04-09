import numpy as np
import cv2

def contour(
    img_gray: np.ndarray,
    G: np.ndarray
) -> np.ndarray:
    """
    Finds contours in the grayscale image and updates the gradient score map G to set pixels within large shadow contours to 1.0.
    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image (uint8 or float32).
        Input image (grayscale uint8 or float32, or BGR/RGB colour).
        Colour images are converted to grayscale automatically.
    G : np.ndarray
        Per-pixel gradient score map (shape (H, W), dtype float32, values ∈ [0, 1]).
    Returns
    -------
    G : np.ndarray
        Updated gradient score map.
    """
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_gray.copy()
 
    img_gray = img_gray.astype(np.float32)
    COLOR_THRESHOLD = 40 # Adjust this threshold based on your images to detect shadows effectively
    _, binary_mask = cv2.threshold(img_gray, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_shadows_mask = np.zeros_like(img_gray)

    MIN_AREA_THRESHOLD = 500  # Adjust this based on your image resolution

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA_THRESHOLD:
            cv2.drawContours(large_shadows_mask, [contour], -1, 255, thickness=cv2.FILLED)

    G[large_shadows_mask == 255] = 1.0
    return G