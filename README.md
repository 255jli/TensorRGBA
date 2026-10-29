# TensorRGBA

Neural network weights stored inside a PNG image.

The trained model lives completely inside one lossless PNG file.  
You load the image in the browser with a pure JavaScript PNG decoder and run inference with pure JavaScript. No backend. No heavy libraries.

## What it does

- Trains a small conditional MLP that generates handwritten digits (MNIST style)
- Packs all weights into a PNG in two formats:
  - **float32** — full precision (1 weight = 1 pixel)
  - **int8** — quantized (4 weights = 1 pixel)
- Restores the model and generates digits entirely on the client

## Project structure

```
train/                 - digit generator (reference example)
web/                   - pure JS loader + inference + demo
docs/
  integration.md       - how to pack your own model
  architecture.md      - design notes
examples/              - future additional models
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
python train.py
```

Default settings (tuned for quick convergence):

- epochs: 15
- step size for LR scheduler: 5
- batch size: 128

Checkpoints are saved to `./checkpoints/`.  
Final model: `checkpoints/generator_final.pt`

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

## Cross-platform loading

The loader never uses Canvas or ImageBitmap for model data.  
It fetches the file as an ArrayBuffer and decodes the PNG with a small pure-JavaScript decoder. This avoids Windows color-profile and gamma issues that destroy binary weights.

## Extending the project

See `docs/integration.md` for a step-by-step guide on packing your own models.

Planned additional examples (see `examples/`):

- Black-and-white animal generator
- Simple image cleanup (watermark / background)
- Experimental tiny language model

## Notes

- Only PNG is supported. JPEG destroys the weights.
- The image looks like noise. That is expected for the reference implementation.
- Architecture is intentionally simple so the pure-JS runtime stays fast and readable.
- No post-processing is applied on the JavaScript side — the network output is shown as-is.

## License

MIT
