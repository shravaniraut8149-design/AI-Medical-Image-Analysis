
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class ChestXRayDataset(Dataset):
    def __init__(self, split):
        self.data = load_dataset(
            "mmenendezg/pneumonia_x_ray"
        )[split]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample["image"].convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label


if __name__ == "__main__":
    train_dataset = ChestXRayDataset("train")
    val_dataset = ChestXRayDataset("validation")
    test_dataset = ChestXRayDataset("test")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("Dataset preparation successful!")
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Class names: NORMAL (0), PNEUMONIA (1)")
print("PREPARE DATA FILE IS RUNNING")

