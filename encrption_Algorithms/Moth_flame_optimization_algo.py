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


def moth_flame_optimization_encryption(image, key):
    """
    Encrypt the image using Moth Flame Optimization logic with a spiral transformation of the key.
    """
    assert image.shape == key.shape, "Key and image must have the same dimensions"

    # Generate a logarithmic spiral pattern for the key
    indices = np.arange(1, key.size + 1).reshape(key.shape)  # Match shape
    spiral_key = key * np.log1p(indices % 255)

    # Normalize the spiral key
    spiral_key = (spiral_key % 255).astype(np.uint8)

    # Apply XOR encryption
    return np.bitwise_xor(image, spiral_key)

def process_images(image_paths, chaotic_map_func, output_folder, bits=40):
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Image not found: {image_path}. Skipping...")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        key_str = chaotic_map_func(bits)
        key = np.array([int(bit) for bit in key_str], dtype=np.uint8) * 255
        key = np.tile(key, (image.size // len(key_str) + 1))[:image.size].reshape(image.shape)

        encrypted_image = moth_flame_optimization_encryption(image, key)
        output_path = os.path.join(output_folder, f"encrypted_{base_name}.png")
        cv2.imwrite(output_path, encrypted_image)
        print(f"Encrypted {base_name} saved at {output_path}")

if __name__ == "__main__":
    image_paths = [
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\lena.png",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\barbara.jpg",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\cameraman.jpg",
    ]
    output_folder = "mfoImages"
    
    # Uncomment the desired chaotic map function
    # chaotic_map_func = logistic_map  # Logistic map
    # chaotic_map_func = sine_map      # Sine map
    # chaotic_map_func = gauss_map     # Gauss map
    chaotic_map_func = circle_map    # Circle map

    process_images(image_paths, chaotic_map_func, output_folder)
