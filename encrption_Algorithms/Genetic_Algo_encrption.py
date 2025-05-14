import numpy as np
import cv2
import os

# Logistic Map
def logistic_map(bits, seed=0.5, r=3.99):
    """
    Generate a key using the logistic map.
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = r * x * (1 - x)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))

# Sine Map
def sine_map(bits, seed=0.5):
    """
    Generate a key using the sine map.
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.sin(np.pi * x)
        chaotic_sequence.append(int(x * 2))
    return "".join(map(str, chaotic_sequence))

# Gauss Map
def gauss_map(bits, seed=0.5):
    """
    Generate a key using the Gauss map.
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.exp(-x ** 2)
        chaotic_sequence.append(int(x * 2))
    return "".join(map(str, chaotic_sequence))

# Circle Map
def circle_map(bits, seed=0.5, alpha=0.5):
    """
    Generate a key using the circle map.
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = (x + alpha) % 1
        chaotic_sequence.append(int(x * 2))
    return "".join(map(str, chaotic_sequence))

def encrypt_image_with_key(image, key):
    """
    Encrypt the image using XOR operation with the provided key.
    """
    return np.bitwise_xor(image, key)

def genetic_algorithm_encryption(image, key):
    """
    Encrypt the image using Genetic Algorithm logic with crossover and mutation on the key.
    """
    # Simulate crossover: Swap first and second half
    midpoint = len(key) // 2
    key = np.concatenate((key[midpoint:], key[:midpoint]))

    # Simulate mutation: Flip ~10% of the bits
    mutation_indices = np.random.choice(len(key), size=len(key) // 10, replace=False)
    key[mutation_indices] = 255 - key[mutation_indices]

    return encrypt_image_with_key(image, key)

def process_images(image_paths, chaotic_map_func, output_folder, bits=40):
    """
    Encrypt multiple images using Genetic Algorithm encryption and save them.
    """
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Image not found: {image_path}. Skipping...")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Generate chaotic key
        key_str = chaotic_map_func(bits)
        key = np.array([int(bit) for bit in key_str], dtype=np.uint8) * 255

        # Resize key to match image size
        key = np.tile(key, (image.size // len(key_str) + 1))[:image.size].reshape(image.shape)

        # Encrypt the image
        encrypted_image = genetic_algorithm_encryption(image, key)

        # Save output
        output_path = os.path.join(output_folder, f"encrypted_{base_name}.png")
        cv2.imwrite(output_path, encrypted_image)
        print(f"Encrypted {base_name} saved at {output_path}")

if __name__ == "__main__":
    # Define image paths (raw strings or forward slashes)
    image_paths = [
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\lena.png",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\barbara.jpg",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\cameraman.jpg",
    ]

    # Output folder
    output_folder = "gaImages"

    # Choose chaotic map
    # chaotic_map_func = logistic_map
    # chaotic_map_func = sine_map
    # chaotic_map_func = gauss_map
    chaotic_map_func = circle_map

    # Run encryption
    process_images(image_paths, chaotic_map_func, output_folder)
