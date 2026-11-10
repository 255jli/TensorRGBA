import argparse
import struct

import numpy as np
import torch
from PIL import Image

from model import CifarGenerator


MAGIC = b"TRGBA001"
HEADER_PIXELS = 16


def collect_weights(model):
    weights = []
    for tensor in model.export_tensors():
        weights.append(tensor.detach().cpu().contiguous().view(-1))
    return torch.cat(weights).numpy().astype(np.float32)


def build_header(mode, noise_dim, num_classes, total_params):
    header = bytearray(HEADER_PIXELS * 4)
    header[0:8] = MAGIC
    header[8] = mode
    struct.pack_into("<H", header, 9, noise_dim)
    header[11] = num_classes
    struct.pack_into("<I", header, 12, total_params)
    header[24] = 1
    return header


def export_float32(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "generator_state_dict" not in ckpt:
        raise ValueError("Checkpoint is not from the conditional GAN trainer")
    noise_dim = ckpt.get("noise_dim", 128)
    num_classes = ckpt.get("num_classes", 2)
    model = CifarGenerator(noise_dim=noise_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()

    flat = collect_weights(model)
    total_params = len(flat)

    header_bytes = build_header(
        mode=0,
        noise_dim=noise_dim,
        num_classes=num_classes,
        total_params=total_params,
    )

    weight_bytes = flat.tobytes()
    all_bytes = header_bytes + weight_bytes

    num_pixels = (len(all_bytes) + 3) // 4
    side = int(np.ceil(np.sqrt(num_pixels)))
    padded_size = side * side * 4
    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[:len(all_bytes)] = np.frombuffer(all_bytes, dtype=np.uint8)

    rgba = padded.reshape(side, side, 4)
    img = Image.fromarray(rgba, "RGBA")
    img.save(output_path, "PNG")
    print(f"Float32 model exported: {output_path}")
    print(f"Image size: {side}x{side}")
    print(f"Total model values: {total_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="../cifar10_float32.png",
    )
    args = parser.parse_args()
    export_float32(args.checkpoint, args.output)
