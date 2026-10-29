# TensorRGBA

Neural network weights stored inside a PNG image.

The trained generator lives completely inside one lossless PNG file.  
You load the image in the browser, extract the weights with a pure JavaScript PNG decoder, and run inference with pure JavaScript. No backend. No heavy libraries.

## What it does

- Trains a small conditional MLP that generates handwritten digits (MNIST style)
- Packs all weights into a PNG in two formats:
  - **float32** — full precision (1 weight = 1 pixel)
  - **int8** — quantized (4 weights = 1 pixel)
- Restores the model and generates digits entirely on the client

## Project structure

```
train/
  model.py           - generator definition
  train.py           - training script
  export_float32.py  - export full precision PNG
  export_int8.py     - export quantized PNG
web/
  png_decoder.js     - pure JS PNG decoder (no Canvas)
  load_model.js      - PNG → weights
  inference.js       - pure JS forward pass
  index.html         - minimal demo
requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Pillow
- numpy

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
cd train
python train.py --epochs 50 --batch-size 128 --lr 0.0002
```

Checkpoints are saved to `./checkpoints/`.  
Final model: `checkpoints/generator_final.pt`

You can run it on CPU. It is not a heavy model.

## Export to PNG

Float32 (full precision):

```bash
python export_float32.py --checkpoint checkpoints/generator_final.pt --output ../model_float32.png
```

Int8 (quantized):

```bash
python export_int8.py --checkpoint checkpoints/generator_final.pt --output ../model_int8.png
```

## Run the demo

Open `web/index.html` in a browser (or serve the folder with any static server).

1. Load the exported PNG
2. Choose a digit
3. Press Generate

## How the packing works

### Float32 mode
Every weight is a 32-bit float.  
4 bytes → one RGBA pixel.

### Int8 mode
Weights are min-max quantized to 0..255.  
4 quantized values → one RGBA pixel.  
Scale and min are stored in the header.

### Header
First 16 pixels (64 bytes) contain:

- Magic bytes `TRGBA001`
- Mode (0 = float32, 1 = int8)
- noise dimension
- number of classes
- total parameter count
- quantization parameters (int8 only)

## Cross-platform loading

Browsers (especially on Windows) apply display color profiles and gamma correction when decoding images through Canvas or ImageBitmap. This changes individual bytes and destroys the packed weights.

The loader therefore never touches Canvas or ImageBitmap for model data.  
It fetches the file as an ArrayBuffer and decodes the PNG with a small pure-JavaScript decoder (`png_decoder.js`) that:

- Parses PNG chunks directly
- Inflates IDAT with `DecompressionStream`
- Reconstructs filtered scanlines
- Returns the exact raw RGBA bytes written by the exporter

This works reliably on Windows, macOS and Linux.

## Notes

- Only PNG is supported. JPEG destroys the weights.
- The image looks like noise. That is expected.
- Architecture is intentionally simple so the pure-JS runtime stays fast and readable.

## License

MIT
