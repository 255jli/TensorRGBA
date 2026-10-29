import argparse
import struct

import numpy as np
import torch
from PIL import Image

from model import AnimalGenerator


MAGIC = b"TRGBA001"
HEADER_PIXELS = 16


def collect_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().contiguous().view(-1))
    return torch.cat(weights).numpy().astype(np.float32)


def build_header(mode, noise_dim, num_classes, total_params):
    header = bytearray(HEADER_PIXELS * 4)
    header[0:8] = MAGIC
    header[8] = mode
    struct.pack_into("<H", header, 9, noise_dim)
    header[11] = num_classes
    struct.pack_into("<I", header, 12, total_params)
    return header


def export_float32(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    noise_dim = ckpt.get("noise_dim", 128)
    model = AnimalGenerator(noise_dim=noise_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    flat = collect_weights(model)
    total_params = len(flat)

    header_bytes = build_header(mode=0, noise_dim=noise_dim, num_classes=2, total_params=total_params)

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
    print(f"Total parameters: {total_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="animal_float32.png")
    args = parser.parse_args()
    export_float32(args.checkpoint, args.output)
