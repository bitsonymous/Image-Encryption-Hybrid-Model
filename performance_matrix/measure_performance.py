import numpy as np
import cv2

def calculate_npcr(original, encrypted):
    """
    Calculate Number of Pixels Change Rate (NPCR).
    """
    diff = original != encrypted
    npcr = np.sum(diff) / diff.size * 100
    return npcr

def calculate_uaci(original, encrypted):
    """
    Calculate Unified Average Changing Intensity (UACI).
    """
    intensity_diff = np.abs(original.astype(np.float64) - encrypted.astype(np.float64))
    uaci = np.mean(intensity_diff / 255) * 100
    return uaci

def calculate_entropy(image):
    """
    Calculate Information Entropy (IE).
    """
    pixel_counts = np.bincount(image.flatten(), minlength=256)
    probabilities = pixel_counts / np.sum(pixel_counts)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)

def process_images(original_paths, encrypted_paths):
    """
    Process and calculate performance metrics for a list of original and encrypted image pairs.
    """
    for original_path, encrypted_path in zip(original_paths, encrypted_paths):
        # Load original and encrypted images
        original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        encrypted_image = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)

        if original_image is None or encrypted_image is None:
            print(f"Error: Failed to load images {original_path} or {encrypted_path}")
            continue

        # Calculate NPCR
        npcr = calculate_npcr(original_image, encrypted_image)

        # Calculate UACI
        uaci = calculate_uaci(original_image, encrypted_image)

        # Calculate Entropy
        entropy_value = calculate_entropy(encrypted_image)

        # Print results
        image_name = original_path.split('/')[-1]
        print(f"Results for {image_name}:")
        print(f"  NPCR: {npcr:.4f}%")
        print(f"  UACI: {uaci:.4f}%")
        print(f"  Entropy: {entropy_value:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    # Paths to original and encrypted images
    original_image_paths = [
        "/home/chamoli/Desktop/minor project/CODE/dataset/lena.png",
        "/home/chamoli/Desktop/minor project/CODE/dataset/cameraman.jpg",
        "/home/chamoli/Desktop/minor project/CODE/dataset/barbara.jpg",
    ]
    encrypted_image_paths = [
        "/home/chamoli/Desktop/minor project/CODE/hybridImages/encrypted_lena.png",
        "/home/chamoli/Desktop/minor project/CODE/hybridImages/encrypted_cameraman.png",
        "/home/chamoli/Desktop/minor project/CODE/hybridImages/encrypted_barbara.png",
    ]

    # Process images and calculate metrics
    process_images(original_image_paths, encrypted_image_paths)
