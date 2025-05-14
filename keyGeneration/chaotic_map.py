import numpy as np

# Logistic Map
def logistic_map(bits, seed=0.5, r=3.99):
    """
    Generate a 40-bit key using the logistic map.
    :param bits: Number of bits for the key (e.g., 40)
    :param seed: Initial value (0 < seed < 1)
    :param r: Control parameter (3.57 < r ≤ 4 for chaos)
    :return: 40-bit key as a binary string
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
    Generate a 40-bit key using the sine map.
    :param bits: Number of bits for the key (e.g., 40)
    :param seed: Initial value (0 < seed < 1)
    :return: 40-bit key as a binary string
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.sin(np.pi * x)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))


# Gauss Map
def gauss_map(bits, seed=0.5):
    """
    Generate a 40-bit key using the Gauss map.
    :param bits: Number of bits for the key (e.g., 40)
    :param seed: Initial value (0 < seed < 1)
    :return: 40-bit key as a binary string
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = np.exp(-x ** 2)
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))


# Circle Map
def circle_map(bits, seed=0.5, alpha=0.5):
    """
    Generate a 40-bit key using the circle map.
    :param bits: Number of bits for the key (e.g., 40)
    :param seed: Initial value (0 < seed < 1)
    :param alpha: Control parameter for the map
    :return: 40-bit key as a binary string
    """
    x = seed
    chaotic_sequence = []
    for _ in range(bits):
        x = (x + alpha) % 1
        chaotic_sequence.append(int(x * 2))  # Normalize to binary (0 or 1)
    return "".join(map(str, chaotic_sequence))


# Example Usage
if __name__ == "__main__":
    bits = 40  # Generate a 40-bit key
    print("Logistic Map Key: ", logistic_map(bits))
    print("Sine Map Key:     ", sine_map(bits))
    print("Gauss Map Key:    ", gauss_map(bits))
    print("Circle Map Key:   ", circle_map(bits))
