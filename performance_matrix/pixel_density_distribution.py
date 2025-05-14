import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def plot_histogram_comparison(original, encrypted):
    """
    Plot histograms and images for original and encrypted images side by side.
    :param original: Original image (numpy array).
    :param encrypted: Encrypted image (numpy array).
    """
    # Create a histogram for the original image
    hist_original, bins_original = np.histogram(original.flatten(), bins=256, range=[0, 256])
    # Create a histogram for the encrypted image
    hist_encrypted, bins_encrypted = np.histogram(encrypted.flatten(), bins=256, range=[0, 256])

    # Create a figure with 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original Image
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Original Image')

    # Original Histogram
    axs[0, 1].plot(bins_original[0:-1], hist_original, color='blue')
    axs[0, 1].set_xlim([0, 256])
    axs[0, 1].set_title('Original Histogram')
    axs[0, 1].set_xlabel('Pixel Intensity')
    axs[0, 1].set_ylabel('Frequency')

    # Encrypted Image
    axs[1, 0].imshow(encrypted, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Encrypted Image')

    # Encrypted Histogram
    axs[1, 1].plot(bins_encrypted[0:-1], hist_encrypted, color='blue')
    axs[1, 1].set_xlim([0, 256])
    axs[1, 1].set_title('Encrypted Histogram')
    axs[1, 1].set_xlabel('Pixel Intensity')
    axs[1, 1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("histogramPlots", exist_ok=True)
    plt.savefig("histogramPlots/histogram_comparison_cameraman_hybrid.png")
    plt.close()

if __name__ == "__main__":
    # Paths to the images
    original_image_path = r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\cameraman.jpg"
    encrypted_image_path = r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\hybridImages\encrypted_cameraman.png"
    
    # Check if the images exist at the given paths
    if not os.path.exists(original_image_path):
        print(f"Original image not found at {original_image_path}")
    if not os.path.exists(encrypted_image_path):
        print(f"Encrypted image not found at {encrypted_image_path}")
    
    # Load the original and encrypted images
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    encrypted_image = cv2.imread(encrypted_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images were successfully loaded
    if original_image is None:
        raise FileNotFoundError(f"Failed to load original image from {original_image_path}")
    if encrypted_image is None:
        raise FileNotFoundError(f"Failed to load encrypted image from {encrypted_image_path}")

    # Plot and save histograms
    plot_histogram_comparison(original_image, encrypted_image)

    print("Histogram comparison saved in 'histogramPlots' folder.")
