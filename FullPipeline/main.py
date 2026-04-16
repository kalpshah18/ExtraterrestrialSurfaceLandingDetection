from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from contour import contour
from gradientScore import gradient_score
from terrainClassifier import classify_surface_fft


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_default_images() -> list[Path]:
    samples_dir = _repo_root() / "GradientScoring" / "Samples"
    if not samples_dir.exists():
        return []

    return [
        path
        for path in sorted(samples_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _load_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, image_gray


def _build_prediction_map(
    gradient_map: np.ndarray,
    terrain_prediction: np.ndarray,
    safe_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the two predictions into safe/unsafe pixel masks.

    Terrain probability is mapped as hard -> 1.0 and soft -> 0.65.
    The final safe score is:
        safe_score = (1 - gradient_risk) * terrain_probability
    Pixels with safe_score >= safe_threshold are marked safe.
    """
    terrain_prediction = terrain_prediction.astype(np.uint8)
    terrain_probability = np.where(terrain_prediction == 1, 1.0, 0.65).astype(np.float32)
    slope_safety = 1.0 - np.clip(gradient_map, 0.0, 1.0)

    safe_score = slope_safety * terrain_probability
    safe_mask = safe_score >= safe_threshold

    # For display: 0 -> safe, 1 -> unsafe.
    prediction_map = np.logical_not(safe_mask).astype(np.uint8)
    return prediction_map, safe_mask.astype(np.uint8), safe_score


def find_large_safe_areas(
    prediction_map: np.ndarray,
    min_area: int = 1500,
    safe_value: int = 0,
) -> np.ndarray:
    """Extract connected safe regions larger than ``min_area``.

    Parameters
    ----------
    prediction_map : np.ndarray
        Binary safe/unsafe map where safe pixels are ``safe_value`` and all
        other pixels are treated as unsafe.
    min_area : int
        Minimum number of pixels in a connected component to keep.
    safe_value : int
        Pixel value that represents safe cells in ``prediction_map``.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) of large safe areas: 1 for kept safe pixels, 0
        otherwise.
    """
    if prediction_map.ndim != 2:
        raise ValueError("prediction_map must be a 2D array")

    safe_binary = (prediction_map == safe_value).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        safe_binary,
        connectivity=8,
    )

    large_safe_mask = np.zeros_like(safe_binary, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            large_safe_mask[labels == label] = 1

    return large_safe_mask


def process_image(
    image_path: Path,
    safe_threshold: float,
    large_safe_min_area: int,
) -> dict[str, np.ndarray | str]:
    image_bgr, image_gray = _load_image(image_path)

    gradient_map = gradient_score(image_bgr, percentile=99.0, ksize=3, blur_ksize=5)
    gradient_map = contour(image_gray, gradient_map.copy())

    terrain_prediction, _ = classify_surface_fft(
        image_gray,
        block_size=32,
        threshold_radius=6,
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction_map, safe_mask, safe_score = _build_prediction_map(
        gradient_map,
        terrain_prediction,
        safe_threshold=safe_threshold,
    )
    large_safe_areas = find_large_safe_areas(
        prediction_map,
        min_area=large_safe_min_area,
        safe_value=0,
    )

    return {
        "name": image_path.name,
        "original": image_rgb,
        "prediction_map": prediction_map,
        "safe_mask": safe_mask,
        "safe_score": safe_score,
        "large_safe_areas": large_safe_areas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Mars terrain pipeline: gradient scoring, contouring, and FFT terrain classification."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=(
            "Optional image paths. If omitted, the script uses the sample images in "
            "GradientScoring/Samples."
        ),
    )
    parser.add_argument(
        "--safe-threshold",
        type=float,
        default=0.35,
        help=(
            "Safe-score threshold used after multiplying terrain probability and "
            "gradient-derived safety probability. Default: 0.35"
        ),
    )
    parser.add_argument(
        "--large-safe-min-area",
        type=int,
        default=1500,
        help=(
            "Minimum connected safe-region area (in pixels) retained in the "
            "large-safe-area map. Default: 1500"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.images:
        image_paths = [Path(image) for image in args.images]
    else:
        image_paths = _discover_default_images()

    if not image_paths:
        print("No input images were found.")
        return 1

    results = []
    for image_path in image_paths:
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            results.append(
                process_image(
                    image_path,
                    safe_threshold=args.safe_threshold,
                    large_safe_min_area=args.large_safe_min_area,
                )
            )
        except ValueError as exc:
            print(exc)

    if not results:
        print("No images were processed successfully.")
        return 1

    output_dir = _repo_root() / "final output"
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_to_save = min(3, len(results))
    for index in range(examples_to_save):
        result = results[index]
        image_stem = Path(result["name"]).stem
        output_path = output_dir / f"{index + 1}_{image_stem}_large_safe_areas.png"

        save_fig, save_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        save_axes[0].imshow(result["original"])
        save_axes[0].set_title(f"Original: {result['name']}")
        save_axes[0].axis("off")

        save_axes[1].imshow(result["large_safe_areas"], cmap="Greens", vmin=0, vmax=1)
        save_axes[1].set_title(
            f"Large safe areas (min area={args.large_safe_min_area})\n"
            "Black: filtered out | Green: large safe area"
        )
        save_axes[1].axis("off")

        save_fig.savefig(output_path, dpi=200)
        plt.close(save_fig)

    print(f"Saved {examples_to_save} example(s) to: {output_dir}")

    num_images = len(results)
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 5 * num_images), constrained_layout=True)

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, result in enumerate(results):
        axes[row_index, 0].imshow(result["original"])
        axes[row_index, 0].set_title(f"Original: {result['name']}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(result["large_safe_areas"], cmap="Greens", vmin=0, vmax=1)
        axes[row_index, 1].set_title(
            f"Large safe areas (min area={args.large_safe_min_area})\n"
            "Black: filtered out | Green: large safe area"
        )
        axes[row_index, 1].axis("off")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())