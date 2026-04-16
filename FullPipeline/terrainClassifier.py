import numpy as np
from scipy.fft import fft2, fftshift


def classify_surface_fft(image, block_size=32, threshold_radius=6, threshold_ratio=1.0):
	"""
	Classify grayscale terrain into soft (0) and hard (1) using FFT block texture.

	The function reflection-pads the image so all edge pixels are processed using
	complete blocks. Outputs are cropped back to the original image size.
	"""
	orig_h, orig_w = image.shape

	# Pad to make image dimensions divisible by block size.
	pad_h = (block_size - (orig_h % block_size)) % block_size
	pad_w = (block_size - (orig_w % block_size)) % block_size
	padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
	h, w = padded_image.shape

	prediction_matrix = np.zeros((h, w), dtype=np.uint8)
	ratio_matrix = np.zeros((h, w), dtype=float)
	stride = block_size

	# Build radial masks around FFT center for low/high frequency split.
	cy, cx = block_size // 2, block_size // 2
	y, x = np.ogrid[:block_size, :block_size]
	dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

	high_freq_mask = dist_from_center > threshold_radius
	low_freq_mask = dist_from_center <= threshold_radius

	ratios = []

	for i in range(0, h, stride):
		for j in range(0, w, stride):
			block = padded_image[i : i + block_size, j : j + block_size]

			f_transform = fft2(block)
			f_shift = fftshift(f_transform)
			magnitude_spectrum = np.abs(f_shift)

			high_energy = np.sum(magnitude_spectrum[high_freq_mask])
			low_energy = np.sum(magnitude_spectrum[low_freq_mask])
			ratio = high_energy / (low_energy + 1e-8)

			ratios.append((i, j, ratio))

	for i, j, ratio in ratios:
		is_hard = 1 if ratio > threshold_ratio else 0
		prediction_matrix[i : i + block_size, j : j + block_size] = is_hard
		ratio_matrix[i : i + block_size, j : j + block_size] = ratio

	final_prediction = prediction_matrix[:orig_h, :orig_w]
	final_ratio = ratio_matrix[:orig_h, :orig_w]

	return final_prediction, final_ratio
