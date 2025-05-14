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