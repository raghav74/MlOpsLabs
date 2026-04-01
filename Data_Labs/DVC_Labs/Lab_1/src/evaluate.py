import json
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from train import SimpleCNN


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    batch_size = params["train"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_labels = torch.load("data/processed/test_labels.pt", weights_only=True)

    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/model.pt", map_location=device, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / total

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss:     {avg_loss:.4f}")

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump({"accuracy": round(accuracy, 4), "loss": round(avg_loss, 4)}, f, indent=2)

    print("Metrics saved to metrics/metrics.json")


if __name__ == "__main__":
    main()
