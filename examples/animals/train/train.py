import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import AnimalGenerator


CAT_CLASS = 3
DOG_CLASS = 5


def one_hot(labels, num_classes=2):
    return torch.nn.functional.one_hot(labels, num_classes).float()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    indices = []
    mapped_labels = []
    for idx, (_, label) in enumerate(full_dataset):
        if label == CAT_CLASS:
            indices.append(idx)
            mapped_labels.append(0)
        elif label == DOG_CLASS:
            indices.append(idx)
            mapped_labels.append(1)

    class AnimalSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices, labels):
            self.base = base
            self.indices = indices
            self.labels = labels

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            img, _ = self.base[self.indices[i]]
            return img, self.labels[i]

    dataset = AnimalSubset(full_dataset, indices, mapped_labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = AnimalGenerator(noise_dim=args.noise_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels_oh = one_hot(torch.tensor(labels)).to(device)
            noise = torch.randn(images.size(0), args.noise_dim, device=device)

            optimizer.zero_grad()
            outputs = model(noise, labels_oh)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = Path(args.output_dir) / f"animal_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "noise_dim": args.noise_dim,
                "epoch": epoch,
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = Path(args.output_dir) / "animal_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "noise_dim": args.noise_dim,
        "epoch": args.epochs,
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--step-size", type=int, default=6)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    train(args)
