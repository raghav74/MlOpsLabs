import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_labels = torch.load("data/processed/train_labels.pt", weights_only=True)

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs} -- loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")


if __name__ == "__main__":
    main()
