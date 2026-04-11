from lib import build_terrain_matrix, save_terrain_overlay

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
    image_path="6.jpg",
    labeled_pixels=labeled_pixels,
    classifier_name="rf",   # or "svm" or "knn"
    stride=4,               # 1 = full resolution
)

# 3. terrain_matrix is an H×W numpy array with values 0, 1, or 2

# 4. Optional: save a colour overlay
save_terrain_overlay("6.jpg", terrain_matrix, "output_overlay.png")