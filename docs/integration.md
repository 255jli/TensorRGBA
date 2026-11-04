# Integrating your model into a PNG

This guide explains how to pack any small PyTorch model into a TensorRGBA PNG and load it in pure JavaScript.

## 1. Export requirements

Your model must be fully defined by a flat list of parameters (weights + biases).  
No dynamic control flow, no external embeddings, no large lookup tables.

Recommended maximum size for comfortable browser use: under 2–4 million parameters (float32).

## 2. Collect weights in a fixed order

```python
def collect_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().contiguous().view(-1))
    return torch.cat(weights).numpy().astype(np.float32)
```

The order of `model.parameters()` must stay identical between export and the JavaScript reconstruction.

## 3. Header format (64 bytes)

| Offset | Size | Content                          |
|--------|------|----------------------------------|
| 0      | 8    | Magic `TRGBA001`                 |
| 8      | 1    | Mode (0 = float32, 1 = int8)     |
| 9      | 2    | Extra dimension / noise dim      |
| 11     | 1    | Number of classes or flags       |
| 12     | 4    | Total parameter count            |
| 16     | 4    | min value (int8 only)            |
| 20     | 4    | scale (int8 only)                |
| 24–63  | 40   | Reserved / future metadata       |

After the header come the raw weight bytes.

## 4. Float32 packing

Each float32 becomes one RGBA pixel (4 bytes).  
Pad the image to a square if needed.

## 5. Int8 packing

Min-max quantize the entire weight tensor to 0–255, store `min` and `scale` in the header, then pack 4 quantized values per pixel.

## 6. JavaScript side

1. Fetch the PNG as `ArrayBuffer`.
2. Decode with the pure JS PNG decoder (`png_decoder.js`) — never use Canvas or ImageBitmap for model data.
3. Read the 64-byte header.
4. Reconstruct `Float32Array` of weights.
5. Rebuild layers in the exact same order used during export.
6. Run forward pass with simple matrix multiplies.

## 7. Keeping inference transparent

Do not add any post-processing, sharpening, or “magic” filters on the JavaScript side.  
The output of the network should be exactly what the user sees. This keeps the demo honest and easy to understand.

## 8. Recommended project layout for new models

```
examples/
  my_model/
    train/
      model.py
      train.py
      export_float32.py
      export_int8.py
    web/
      load_model.js
      inference.js
      index.html
```

Copy the MNIST example in `examples/mnist/` and replace the architecture and
data loading.

## 9. What works well today

- Small MLPs
- Tiny CNNs (if you implement the corresponding convolutions in JS)
- Autoencoders / simple generators
- Small policy networks for games

## 10. What is still hard

- Large language models (size + attention complexity)
- Models that need heavy normalization or complex ops
- Anything that relies on large external embeddings

Start small, keep the JavaScript side simple, and the PNG format stays reliable across platforms.
