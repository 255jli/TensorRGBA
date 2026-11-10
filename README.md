# TensorRGBA

Neural network weights stored inside a PNG image.

The trained model lives completely inside one lossless PNG file.  
You load the image in the browser with a pure JavaScript PNG decoder and run inference with pure JavaScript. No backend. No heavy libraries.

## What it does

- Packs neural network weights into a PNG in two formats:
  - **float32** — full precision (1 weight = 1 pixel)
  - **int8** — quantized (4 weights = 1 pixel)
- Runs inference entirely on the client with pure JavaScript

## Examples

| Example | Description | Location |
|---------|-------------|----------|
| Digits | Handwritten digit generator (MNIST) | `examples/mnist/` |
| CIFAR-10 | Conditional RGB generator for airplanes and ships | `examples/cifar10/` |

## Project structure

```
examples/
  mnist/                - handwritten digit generator and sample model PNGs
  cifar10/              - conditional airplane/ship RGB GAN
docs/
  integration.md       - how to pack your own model
  architecture.md
requirements.txt
```

## Quick start (digits)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd examples/mnist/train
python train.py
python export_float32.py --checkpoint checkpoints/generator_final.pt --output ../model_float32.png
python export_int8.py --checkpoint checkpoints/generator_final.pt --output ../model_int8.png
```

Open `examples/mnist/web/index.html` and load one of the model PNGs from
`examples/mnist/`.

## CIFAR-10 airplane and ship GAN

```bash
cd examples/cifar10/train
python train.py
python export_float32.py --checkpoint checkpoints/airplane_ship_gan/airplane_ship_gan_final.pt --output ../cifar10_float32.png
python export_int8.py --checkpoint checkpoints/airplane_ship_gan/airplane_ship_gan_final.pt --output ../cifar10_int8.png
```

The generated PNGs are stored in `examples/cifar10/` and can be opened in the browser demo from `examples/cifar10/web/index.html`.

## Download generated model PNGs

The project includes ready-to-open model files in the example folders:

- `examples/mnist/model_float32.png`
- `examples/mnist/model_int8.png`
- `examples/cifar10/cifar10_float32.png`
- `examples/cifar10/cifar10_int8.png`

Open one of those files in the matching demo page, or serve the project locally:

```bash
cd /workspaces/TensorRGBA
python -m http.server 8000
```

Then open:

- `http://localhost:8000/examples/mnist/web/index.html`
- `http://localhost:8000/examples/cifar10/web/index.html`

The generated images are intentionally simple and may look soft; this example is meant to show PNG packing and browser inference clearly, not to produce photorealistic output.

## Cross-platform loading

The loader never uses Canvas or ImageBitmap for model data.  
It fetches the file as an ArrayBuffer and decodes the PNG with a pure-JavaScript decoder. This avoids Windows color-profile and gamma issues.

## Extending

See `docs/integration.md` for a guide on packing your own models.

## Notes

- Only PNG is supported. JPEG destroys the weights.
- The packed image looks like noise — that is expected.
- No post-processing is applied on the JavaScript side.

## License

MIT
