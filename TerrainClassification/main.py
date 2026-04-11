import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import cv2
import os
import sys

def classify_surface_fft(image, block_size=32, threshold_radius=6):
    """
    Classifies a grayscale image into soft (0) and hard (1) surfaces using FFT.
    """
    h, w = image.shape
    prediction_matrix = np.zeros((h, w))
    ratio_matrix = np.zeros((h, w), dtype=float)
    stride = block_size # Non-overlapping for efficiency
    
    # 1. Create spatial masks for Low vs High frequency
    cy, cx = block_size // 2, block_size // 2
    y, x = np.ogrid[:block_size, :block_size]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    high_freq_mask = dist_from_center > threshold_radius
    low_freq_mask = dist_from_center <= threshold_radius
    
    ratios = []
    
    print(f"Processing image of size {w}x{h} in {block_size}x{block_size} blocks...")
    
    # 2. Process the image in blocks
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            block = image[i:i+block_size, j:j+block_size]
            
            # Skip incomplete blocks at the edges
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue
            
            # Compute 2D Fast Fourier Transform
            f_transform = fft2(block)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate energy in high vs low frequency bands
            high_energy = np.sum(magnitude_spectrum[high_freq_mask])
            low_energy = np.sum(magnitude_spectrum[low_freq_mask])
            
            # Calculate Ratio (Add a tiny number to avoid division by zero)
            ratio = high_energy / (low_energy + 1e-8)
            ratios.append({'i': i, 'j': j, 'ratio': ratio})

    # 3. Determine threshold
    # Using the median ratio to dynamically split this specific image's features.
    all_ratios = [r['ratio'] for r in ratios]
    threshold_ratio = np.median(all_ratios) 
    print(f"Calculated dynamic threshold ratio: {threshold_ratio:.4f}")
    
    # 4. Populate the final prediction matrix
    for r in ratios:
        is_hard = 1 if r['ratio'] > threshold_ratio else 0
        i, j = r['i'], r['j']
        
        # Broadcast the prediction to all pixels inside that block
        prediction_matrix[i:i+block_size, j:j+block_size] = is_hard
        ratio_matrix[i:i+block_size, j:j+block_size] = r['ratio']
        
    return prediction_matrix, ratio_matrix

def main():
    image_path = 'testImage.jpg'
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find '{image_path}' in the current directory.")
        sys.exit(1)
        
    # Load image in grayscale
    # FFT texture analysis relies on intensity, not color
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Failed to load '{image_path}'. Ensure it is a valid image file.")
        sys.exit(1)

    # Run the classification
    # You can tweak block_size (e.g., 16, 32, 64) and threshold_radius (e.g., 4, 6, 10)
    prediction_matrix, ratio_matrix = classify_surface_fft(img, block_size=32, threshold_radius=6)
    
    # --- Visualization ---
    print("Generating visualizations...")
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Original Image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Satellite View ('testImage.jpg')")
    axs[0].axis('off')

    # 2. Prediction Matrix
    # Using 'coolwarm' colormap: Blue represents 0 (Soft), Red represents 1 (Hard)
    axs[1].imshow(prediction_matrix, cmap='coolwarm')
    axs[1].set_title("Prediction Matrix\n(Blue: Soft Sand/Dust | Red: Hard Rock)")
    axs[1].axis('off')

    # 3. Overlay
    # Plot original grayscale first, then overlay the prediction matrix with transparency (alpha)
    axs[2].imshow(img, cmap='gray')
    axs[2].imshow(prediction_matrix, cmap='coolwarm', alpha=0.4) 
    axs[2].set_title("Overlay\n(Predictions over Terrain)")
    axs[2].axis('off')

    # 4. FFT Ratio Heatmap
    ratio_plot = axs[3].imshow(ratio_matrix, cmap='inferno')
    axs[3].set_title("FFT Ratio Heatmap\n(High / Low Frequency Energy)")
    axs[3].axis('off')
    fig.colorbar(ratio_plot, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # Save the output figure
    output_filename = 'prediction_output.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully saved visualization to '{output_filename}'")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    main()