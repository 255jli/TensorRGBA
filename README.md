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
| CIFAR-10 | Grayscale generator for all 10 CIFAR-10 classes | `examples/cifar10/` |

## Project structure

```
examples/
  mnist/                - handwritten digit generator and sample model PNGs
  cifar10/              - CIFAR-10 generator
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

## CIFAR-10 example

```bash
cd examples/cifar10/train
python train.py
python export_float32.py --checkpoint checkpoints/animal_final.pt --output ../animal_float32.png
python export_int8.py --checkpoint checkpoints/animal_final.pt --output ../animal_int8.png
```

Open `examples/cifar10/web/index.html`.

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
