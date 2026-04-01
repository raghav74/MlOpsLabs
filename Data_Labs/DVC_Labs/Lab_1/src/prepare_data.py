import os
import yaml
import torch
import numpy as np
from torchvision import datasets, transforms


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    seed = params["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("data/processed", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="data/raw", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data/raw", train=False, download=True, transform=transform
    )

    train_images = torch.stack([img for img, _ in train_dataset])
    train_labels = torch.tensor([label for _, label in train_dataset])
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])

    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(train_labels, "data/processed/train_labels.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(test_labels, "data/processed/test_labels.pt")

    print(f"Train set: {train_images.shape[0]} samples")
    print(f"Test set:  {test_images.shape[0]} samples")
    print("Data saved to data/processed/")


if __name__ == "__main__":
    main()
