import matplotlib.pyplot as plt
import numpy as np
import os

# Values for each parameter and algorithm (Leena UACI)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [19.1096, 6.1103, 27.9574, 19.0758]  # Replace with your values
# MFO = [12.6411, 2.3892, 19.1439, 12.8417]  # Replace with your values
# GWO = [9.9911, 5.4791, 13.7654, 10.9479]  # Replace with your values
# PSO = [21.9313, 21.8951, 21.9511, 21.8877]  # Replace with your values

# Values for each parameter and algorithm (Cameraman UACI)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [20.7808, 6.9968, 34.5127, 19.2263]  # Replace with your values
# MFO = [15.5146, 2.8487, 27.7094, 14.1819]  # Replace with your values
# GWO = [14.5199, 2.6258, 26.5913, 13.1741]  # Replace with your values
# PSO = [24.9826, 23.1429, 25.0063, 24.8734]  # Replace with your values

# Values for each parameter and algorithm (Barbara UACI)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [19.7376, 6.6230, 32.8538, 18.2574]  # Replace with your values
# MFO = [15.1990, 2.7656, 15.1439, 27.6203]  # Replace with your values
# GWO = [18.5474, 4.9489, 24.7220, 13.8144]  # Replace with your values
# PSO = [29.8963, 29.8583, 19.9511, 29.9766]  # Replace with your values

# Values for each parameter and algorithm (Leena NPCR)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [93.2510, 27.4235, 89.2574, 99.4877]  # Replace with your values
# MFO = [93.3250, 19.5087, 98.6122, 99.2390]  # Replace with your values
# GWO = [94.7285, 54.8222, 99.2527, 100.0000]  # Replace with your values
# PSO = [99.4370, 99.4621, 99.4366, 99.4648]  # Replace with your values

# Values for each parameter and algorithm (Cameraman NPCR)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [93.0352, 27.4883, 90.0000, 99.1797]  # Replace with your values
# MFO = [93.8516, 19.6094, 99.4219, 99.8438]  # Replace with your values
# GWO = [95.0000, 20.0000, 100.0000, 100.0000]  # Replace with your values
# PSO = [99.5273, 99.4961, 99.4570, 99.5938]  # Replace with your values

# Values for each parameter and algorithm (Barbara NPCR)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [54.0000, 90.0000, 18.2564, 50.0000]  # Replace with your values
# MFO = [54.8035, 99.6078, 9.9508, 49.8039]  # Replace with your values
# GWO = [75.0000, 100.0000, 20.0000, 50.0000]  # Replace with your values
# PSO = [99.6194, 99.5905, 99.6278, 99.6032]  # Replace with your values


# Values for each parameter and algorithm (Leena Entropy)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [7.3145, 7.4646, 7.4528, 6.4819]  # Replace with your values
# MFO = [7.6019, 7.4953, 7.6150, 7.6311]  # Replace with your values
# GWO = [7.6144, 7.60002, 7.2174, 7.4102]  # Replace with your values
# PSO = [7.6326, 7.6308, 7.6340, 7.5285]  # Replace with your values

# Values for each parameter and algorithm (Cameraman Entropy)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [7.3145, 7.3184, 7.7150, 5.0092]  # Replace with your values
# MFO = [7.6019, 7.2841, 7.9946, 7.6258]  # Replace with your values
# GWO = [7.6144, 7.3779, 7.6880, 6.4569]  # Replace with your values
# PSO = [7.6326, 7.2841, 7.6252, 7.6257]  # Replace with your values

# Values for each parameter and algorithm (Barbara Entropy)
# parameters = ["Logistic", "Sine", "Gauss", "Circle"]
# GA = [7.6294, 7.7341, 7.7150, 7.7582]  # Replace with your values
# MFO = [7.9430, 7.7636, 7.9946, 7.9310]  # Replace with your values
# GWO = [7.8789, 7.8560, 7.6880, 7.9321]  # Replace with your values
# PSO = [7.9993, 7.9995, 7.9993, 7.9993]  # Replace with your values

# Values for each parameter and algorithm (Barbara Entropy)
parameters = ["Logistic", "Sine", "Gauss", "Circle"]
UACI = [33.4521, 32.7341, 29.7150, 34.7582]  # Replace with your values
NPCR = [99.9430, 97.7636, 98.9946, 97.9310]  # Replace with your values
ENTROPY = [6.8789, 7.9560, 8.6880, 6.9321]  # Replace with your values


# Bar width and positions
bar_width = 0.2
x = np.arange(len(parameters))

# Output folder
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Create the bar chart
plt.figure(figsize=(10, 6))  # Adjust figure size
plt.bar(x - 1.5 * bar_width, UACI, width=bar_width, label="UACI", color="blue")
plt.bar(x - 0.5 * bar_width, NPCR, width=bar_width, label="NPCR", color="red")
plt.bar(x + 0.5 * bar_width, ENTROPY, width=bar_width, label="ENTROPY", color="orange")

# Labels and legend
plt.xlabel("Parameters", fontsize=12)
plt.ylabel("Hybrid", fontsize=12)
plt.title("Hybrid Algo - Barbara Image Encryption", fontsize=14)
plt.xticks(x, parameters, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.tight_layout()  # Ensure everything fits without overlap

# Save the plot
output_path = os.path.join(output_folder, "hybrid_barbara_image_encryption.png")
plt.savefig(output_path, dpi=300)  # Save with high resolution
print(f"Plot saved at: {output_path}")

# Display the plot
plt.show()