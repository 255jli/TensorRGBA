# MNIST Classifier

Digit classifier packed into a PNG. Draw a digit in the browser and the network predicts the class.

This example shows the opposite direction of the generators:

- Generator: noise + label → image
- Classifier: image → label probabilities

## Train

```bash
cd examples/classifier/train
python train.py
```

Default: 8 epochs. Typical test accuracy is above 97%.

## Export

```bash
python export_float32.py --checkpoint checkpoints/classifier_final.pt --output ../classifier_float32.png
python export_int8.py --checkpoint checkpoints/classifier_final.pt --output ../classifier_int8.png
```

## Demo

Open `web/index.html`, load one of the exported PNGs, draw a digit, and press Classify.

## Notes

- Input: 28×28 grayscale
- Output: 10 class probabilities
- Architecture: 784 → 256 → 128 → 10
- Same TensorRGBA format and pure JS decoder as the other examples
