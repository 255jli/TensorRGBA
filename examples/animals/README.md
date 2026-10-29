# Animals example

Black-and-white cat and dog generator packed into a PNG.

Uses CIFAR-10 (cat and dog classes only), converted to grayscale 32×32.

## Train

```bash
cd examples/animals/train
python train.py
```

Default: 20 epochs, batch size 64.

## Export

```bash
python export_float32.py --checkpoint checkpoints/animal_final.pt --output ../animal_float32.png
python export_int8.py --checkpoint checkpoints/animal_final.pt --output ../animal_int8.png
```

## Demo

Open `web/index.html`, load the exported PNG, choose Cat or Dog, generate.

## Notes

- Output size: 32×32 grayscale
- Two classes: 0 = cat, 1 = dog
- Same TensorRGBA format and pure JS decoder as the digit example
