import numpy as np
import cv2
import os

# Logistic Map
def logistic_map(bits, seed=0.5, r=3.99):
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = r * x * (1 - x)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))

# Sine Map
def sine_map(bits, seed=0.5):
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.sin(np.pi * x)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))

# Gauss Map
def gauss_map(bits, seed=0.5):
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.exp(-x ** 2)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))

# Circle Map
def circle_map(bits, seed=0.5, alpha=0.5):
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = (x + alpha) % 1
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))

def encrypt_image_with_key(image, key):
    """
    Encrypt the image using XOR operation with the provided key.
    """
    return np.bitwise_xor(image, key)

def grey_wolf_optimization_encryption(image, key):
    """
    Encrypt the image using Grey Wolf Optimization logic with a hierarchical transformation of the key.
    """
    assert key.shape == image.shape, "Key and image must have the same dimensions"

    # Split the key into three groups: alpha, beta, delta
    alpha, beta, delta = np.array_split(key, 3, axis=0)

    # Average influence of alpha, beta, delta
    min_length = min(alpha.shape[0], beta.shape[0], delta.shape[0])
    averaged_key = (alpha[:min_length] + beta[:min_length] + delta[:min_length]) // 3

    # Pad if needed
    padding_needed = image.shape[0] - averaged_key.shape[0]
    if padding_needed > 0:
        pad_width = [(0, padding_needed)] + [(0, 0)] * (len(image.shape) - 1)
        averaged_key = np.pad(averaged_key, pad_width, mode="wrap")

    return encrypt_image_with_key(image, averaged_key)

def process_images(image_paths, chaotic_map_func, output_folder, bits=40):
    """
    Encrypt multiple images using Grey Wolf Optimization encryption and save them.
    """
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Image not found: {image_path}. Skipping...")
            continue

        # Get the base name of the file (e.g., "lena" from "lena.png")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Generate a key using the selected chaotic map
        key_str = chaotic_map_func(bits)
        key = np.array([int(bit) for bit in key_str], dtype=np.uint8) * 255  # Scale to 0-255

        # Tile the key to match the image size
        key = np.tile(key, (image.size // len(key_str) + 1))[:image.size].reshape(image.shape)

        # Encrypt the image
        encrypted_image = grey_wolf_optimization_encryption(image, key)

        # Save the encrypted image with an appropriate name
        output_path = os.path.join(output_folder, f"encrypted_{base_name}.png")
        cv2.imwrite(output_path, encrypted_image)
        print(f"Encrypted {base_name} saved at {output_path}")

if __name__ == "__main__":
    # Define paths to the images
    image_paths = [
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\lena.png",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\barbara.jpg",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\cameraman.jpg",
    ]

    # Define the output folder
    output_folder = "gwoImages"

    # Uncomment the desired chaotic map function
    # chaotic_map_func = logistic_map  # Logistic map
    # chaotic_map_func = sine_map      # Sine map
    # chaotic_map_func = gauss_map     # Gauss map
    chaotic_map_func = circle_map    # Circle map

    # Encrypt and save the images
    process_images(image_paths, chaotic_map_func, output_folder)
