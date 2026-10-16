
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from prepare_data import ChestXRayDataset
from model import ChestXRayCNN
from torch.utils.data import DataLoader
from pathlib import Path


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    # Use Apple Silicon GPU when available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_dataset = ChestXRayDataset("train")
    val_dataset = ChestXRayDataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = ChestXRayCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    model_path = MODEL_DIR / "chest_xray_cnn.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Validation Loss: {val_loss:.4f} "
            f"| Validation Accuracy: {val_accuracy * 100:.2f}%"
        )

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("Best model saved:", model_path)

    # Save training graphs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, epochs + 1),
        [accuracy * 100 for accuracy in val_accuracies],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("CNN Validation Accuracy")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_accuracy.png")
    plt.close()

    print("\nTraining complete!")
    print("Model saved in:", model_path)
    print("Graphs saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
