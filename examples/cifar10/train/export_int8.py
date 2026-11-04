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


def build_header(mode, noise_dim, num_classes, total_params, min_val, scale):
    header = bytearray(HEADER_PIXELS * 4)
    header[0:8] = MAGIC
    header[8] = mode
    struct.pack_into("<H", header, 9, noise_dim)
    header[11] = num_classes
    struct.pack_into("<I", header, 12, total_params)
    struct.pack_into("<f", header, 16, min_val)
    struct.pack_into("<f", header, 20, scale)
    return header


def quantize(weights):
    min_val = float(weights.min())
    max_val = float(weights.max())
    if max_val == min_val:
        scale = 1.0
        quantized = np.zeros_like(weights, dtype=np.uint8)
    else:
        scale = (max_val - min_val) / 255.0
        scaled_weights = (weights - min_val) / scale
        quantized = np.round(scaled_weights).clip(0, 255).astype(np.uint8)
    return quantized, min_val, scale


def export_int8(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    noise_dim = ckpt.get("noise_dim", 128)
    num_classes = ckpt.get("num_classes", 2)
    model = AnimalGenerator(noise_dim=noise_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    flat = collect_weights(model)
    total_params = len(flat)

    quantized, min_val, scale = quantize(flat)

    header_bytes = build_header(
        mode=1,
        noise_dim=noise_dim,
        num_classes=num_classes,
        total_params=total_params,
        min_val=min_val,
        scale=scale,
    )

    weight_bytes = quantized.tobytes()
    all_bytes = header_bytes + weight_bytes

    num_pixels = (len(all_bytes) + 3) // 4
    side = int(np.ceil(np.sqrt(num_pixels)))
    padded_size = side * side * 4
    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[:len(all_bytes)] = np.frombuffer(all_bytes, dtype=np.uint8)

    rgba = padded.reshape(side, side, 4)
    img = Image.fromarray(rgba, "RGBA")
    img.save(output_path, "PNG")
    print(f"Int8 model exported: {output_path}")
    print(f"Image size: {side}x{side}")
    print(f"Total parameters: {total_params}")
    print(f"Quantization min={min_val:.6f} scale={scale:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="animal_int8.png")
    args = parser.parse_args()
    export_int8(args.checkpoint, args.output)
