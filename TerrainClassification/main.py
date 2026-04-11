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
    THRESHOLD_RATIO = 1 # You can adjust this based on experimentation
    print(f"Using static threshold ratio: {THRESHOLD_RATIO:.4f}")
    
    # 4. Populate the final prediction matrix
    for r in ratios:
        is_hard = 1 if r['ratio'] > THRESHOLD_RATIO else 0
        i, j = r['i'], r['j']
        
        # Broadcast the prediction to all pixels inside that block
        prediction_matrix[i:i+block_size, j:j+block_size] = is_hard
        ratio_matrix[i:i+block_size, j:j+block_size] = r['ratio']
        
    return prediction_matrix, ratio_matrix

def main():
    image_files = ['testImage1.jpg', 'testImage2.jpg', 'testImage3.jpg']
    block_size = 32
    threshold_radius = 6
    
    # Store results for all images
    results = []
    
    # Process each image
    for image_path in image_files:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Could not find '{image_path}' in the current directory.")
            continue
            
        # Load image in grayscale
        # FFT texture analysis relies on intensity, not color
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Failed to load '{image_path}'. Ensure it is a valid image file.")
            continue

        print(f"\nProcessing {image_path}...")
        # Run the classification
        prediction_matrix, ratio_matrix = classify_surface_fft(img, block_size=block_size, threshold_radius=threshold_radius)
        
        results.append({
            'filename': image_path,
            'image': img,
            'prediction': prediction_matrix,
            'ratio': ratio_matrix
        })
    
    if not results:
        print("Error: No images were processed successfully.")
        sys.exit(1)

    # --- Visualization: Stack all predictions in a single file ---
    print("\nGenerating stacked visualizations...")
    num_images = len(results)
    fig, axs = plt.subplots(num_images, 4, figsize=(24, 6 * num_images))
    
    # If only one image, axs will be 1D; make it 2D for consistency
    if num_images == 1:
        axs = axs.reshape(1, -1)
    
    for idx, result in enumerate(results):
        img = result['image']
        prediction_matrix = result['prediction']
        ratio_matrix = result['ratio']
        filename = result['filename']
        
        # 1. Original Image
        axs[idx, 0].imshow(img, cmap='gray')
        axs[idx, 0].set_title(f"Original: {filename}")
        axs[idx, 0].axis('off')

        # 2. Prediction Matrix
        # Using 'coolwarm' colormap: Blue represents 0 (Soft), Red represents 1 (Hard)
        axs[idx, 1].imshow(prediction_matrix, cmap='coolwarm')
        axs[idx, 1].set_title(f"Prediction: {filename}\n(Blue: Soft | Red: Hard)")
        axs[idx, 1].axis('off')

        # 3. Overlay
        # Plot original grayscale first, then overlay the prediction matrix with transparency (alpha)
        axs[idx, 2].imshow(img, cmap='gray')
        axs[idx, 2].imshow(prediction_matrix, cmap='coolwarm', alpha=0.4) 
        axs[idx, 2].set_title(f"Overlay: {filename}")
        axs[idx, 2].axis('off')

        # 4. FFT Ratio Heatmap
        ratio_plot = axs[idx, 3].imshow(ratio_matrix, cmap='inferno')
        axs[idx, 3].set_title(f"FFT Ratio: {filename}")
        axs[idx, 3].axis('off')
        fig.colorbar(ratio_plot, ax=axs[idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # Save the output figure
    output_filename = 'prediction_output_stacked.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully saved stacked visualization to '{output_filename}'")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    main()