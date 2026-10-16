
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path

from model import ChestXRayCNN
from prepare_data import ChestXRayDataset


def main():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "chest_xray_cnn.pth"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    test_dataset = ChestXRayDataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = ChestXRayCNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    print("\nTest Accuracy:", f"{accuracy_score(all_labels, all_predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print("Rows = actual labels; columns = predicted labels")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
