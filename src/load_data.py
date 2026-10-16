
from datasets import load_dataset

print("Loading chest X-ray dataset...")

ds = load_dataset("mmenendezg/pneumonia_x_ray")

print("\nDataset loaded successfully!")
print(ds)
print("\nTraining images:", len(ds["train"]))
print("Validation images:", len(ds["validation"]))
print("Test images:", len(ds["test"]))
print("\nClass labels:", ds["train"].features["label"].names)
