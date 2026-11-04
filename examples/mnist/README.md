# MNIST example

Handwritten digit generator with model weights packed into PNG files.

## Train

From the repository root:

```bash
cd examples/mnist/train
python train.py
```

The default run trains for 15 epochs and saves checkpoints locally.

## Export

From `examples/mnist/train`:

```bash
python export_float32.py --checkpoint checkpoints/generator_final.pt --output ../model_float32.png
python export_int8.py --checkpoint checkpoints/generator_final.pt --output ../model_int8.png
```

The example includes float32 and int8 PNG model files in this directory.

## Demo

Open `web/index.html`, load either model PNG, choose a digit, and generate.

## Notes

- Output size: 28×28 grayscale
- Classes: digits 0 through 9
- The browser runs the model directly without image post-processing