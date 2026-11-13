import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CifarDiscriminator, CifarGenerator, initialize_weights


AIRPLANE_CLASS = 0
SHIP_CLASS = 8
NUM_CLASSES = 2
CLASS_NAMES = ["Airplane", "Ship"]


def save_checkpoint(path, generator, args, epoch, generator_loss):
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "noise_dim": args.noise_dim,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epoch": epoch,
        "generator_loss": generator_loss,
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    full_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    indices = [
        index
        for index, label in enumerate(full_dataset.targets)
        if label in (AIRPLANE_CLASS, SHIP_CLASS)
    ]
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    generator = CifarGenerator(args.noise_dim, NUM_CLASSES).to(device)
    discriminator = CifarDiscriminator(NUM_CLASSES).to(device)
    initialize_weights(generator)
    initialize_weights(discriminator)

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        for images, original_labels in loader:
            images = images.to(device)
            labels = (original_labels == SHIP_CLASS).long().to(device)
            batch_size = images.size(0)
            real_targets = torch.full((batch_size, 1), 0.9, device=device)
            fake_targets = torch.zeros((batch_size, 1), device=device)

            noise = torch.rand(
                batch_size,
                args.noise_dim,
                device=device,
            ) * 2 - 1
            fake_images = generator(noise, labels)
            real_scores = discriminator(images, labels)
            fake_scores = discriminator(fake_images.detach(), labels)
            discriminator_loss = (
                criterion(real_scores, real_targets)
                + criterion(fake_scores, fake_targets)
            ) / 2

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            noise = torch.rand(
                batch_size,
                args.noise_dim,
                device=device,
            ) * 2 - 1
            fake_images = generator(noise, labels)
            fake_scores = discriminator(fake_images, labels)
            generator_loss = criterion(
                fake_scores,
                torch.ones_like(fake_scores),
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            total_generator_loss += generator_loss.item()
            total_discriminator_loss += discriminator_loss.item()

        average_generator_loss = total_generator_loss / len(loader)
        average_discriminator_loss = total_discriminator_loss / len(loader)
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"D: {average_discriminator_loss:.4f}  "
            f"G: {average_generator_loss:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_name = f"airplane_ship_gan_epoch_{epoch}.pt"
            checkpoint_path = Path(args.output_dir) / checkpoint_name
            save_checkpoint(
                checkpoint_path,
                generator,
                args,
                epoch,
                average_generator_loss,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = Path(args.output_dir) / "airplane_ship_gan_final.pt"
    save_checkpoint(
        final_path,
        generator,
        args,
        args.epochs,
        average_generator_loss,
    )
    print(f"Training complete. Final generator: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/airplane_ship_gan",
    )
    parser.add_argument("--save-every", type=int, default=10)
    train(parser.parse_args())