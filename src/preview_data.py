
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset
ds = load_dataset("mmenendezg/pneumonia_x_ray")

# Display 6 training images
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    sample = ds["train"][i]
    image = sample["image"]
    label = ds["train"].features["label"].names[sample["label"]]

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label.upper()}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/dataset_preview.png", dpi=150)
plt.show()

print("Preview saved to results/dataset_preview.png")

