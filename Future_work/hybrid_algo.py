import numpy as np
import cv2
import os

def hybrid_algorithm_encryption(image, key):
    """
    Encrypt the image using a hybrid algorithm combining Moth Flame Optimization (MFO)
    and Whale Optimization Algorithm (WOA).
    """
    assert image.shape == key.shape, "Key and image must have the same dimensions"
    
    # Step 1: Apply MFO's spiral transformation
    indices = np.arange(1, key.size + 1).reshape(key.shape)  # Match 3D shape
    spiral_key = key * np.log1p(indices % 255)  # MFO-like transformation
    spiral_key = (spiral_key % 255).astype(np.uint8)  # Normalize to 0-255
    
    # Step 2: Apply WOA's bubble-net strategy
    a = 2  # Convergence control parameter
    for t in range(5):  # Perform 5 iterations for key refinement
        r = np.random.rand(*key.shape)  # Random exploration factor
        A = 2 * a * r - a  # Adaptive control parameter
        C = 2 * r  # Encircling prey
        p = np.random.rand(*key.shape)
        l = np.random.uniform(-1, 1, size=key.shape)  # Spiral factor
        D = np.abs(C * spiral_key - key)
        key = np.where(p < 0.5,  # Bubble-net hunting (exploration vs exploitation)
                       spiral_key - A * D,
                       D * np.exp(a * l) * np.cos(2 * np.pi * l) + spiral_key)
        key = (key % 255).astype(np.uint8)  # Normalize key to the range [0, 255]
        a -= 0.4 / 5  # Linearly decrease `a` over iterations
    
    return np.bitwise_xor(image, key)  # XOR encryption with the hybrid key

def process_images(image_paths, key_str, output_folder):
    """
    Encrypt multiple images using the hybrid algorithm and save them.
    """
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Image not found: {image_path}. Skipping...")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        key = np.array([int(bit) for bit in key_str], dtype=np.uint8) * 255  # Scale to 0-255
        key = np.tile(key, (image.size // len(key_str) + 1))[:image.size].reshape(image.shape)

        encrypted_image = hybrid_algorithm_encryption(image, key)
        output_path = os.path.join(output_folder, f"encrypted_{base_name}.png")
        cv2.imwrite(output_path, encrypted_image)
        print(f"Encrypted {base_name} saved at {output_path}")

if __name__ == "__main__":
    image_paths = [
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\lena.png",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\barbara.jpg",
        r"C:\Users\Himanshu\OneDrive\Desktop\Python\projects\image_enc_metaherustic_llm\dataset\cameraman.jpg",
    ]
    output_folder = "hybridImages"

    def logistic_map(bits, seed=0.5, r=3.99):
        x = seed
        chaotic_sequence = []
        for _ in range(bits):
            x = r * x * (1 - x)
            chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
        return "".join(map(str, chaotic_sequence))
    
    key_str = logistic_map(40)
    process_images(image_paths, key_str, output_folder)
