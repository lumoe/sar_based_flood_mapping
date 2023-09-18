import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open("roc_and_dice_data.pkl", "rb") as f:
    data = pickle.load(f)


avg_dice_coefficients = data["dice_coeffs"]
thresholds = data["thresholds"]

print(len(avg_dice_coefficients))
print(len(thresholds))

optimal_threshold = thresholds[np.argmax(avg_dice_coefficients)]

# Plot ROC curve using average Dice coefficients
plt.figure()
plt.plot(thresholds, avg_dice_coefficients, label="Average Dice Coefficient")
# Add vertical line for optimal threshold with dashed line style and a label
plt.axvline(
    optimal_threshold,
    linestyle="--",
    label=f"Optimal Threshold ({optimal_threshold:.2f})",
    color="orange",
)
plt.xlabel("Threshold")
plt.ylabel("Average Dice Coefficient")
plt.title("ROC Curve")
plt.legend()
plt.show()
plt.savefig("roc_curve.pdf")
