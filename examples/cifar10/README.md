# CIFAR-10 example

Black-and-white CIFAR-10 class-conditional generator packed into a PNG.

Uses all 10 CIFAR-10 classes, converted to grayscale 32×32. The model remains a
fully connected pixel generator; it is not a DCGAN.

## Train

```bash
cd examples/cifar10/train
python train.py
```

Default: 20 epochs, batch size 64.

## Export

```bash
python export_float32.py --checkpoint checkpoints/animal_final.pt --output ../animal_float32.png
python export_int8.py --checkpoint checkpoints/animal_final.pt --output ../animal_int8.png
```

## Demo

Open `web/index.html`, load the exported PNG, choose a CIFAR-10 class, and generate.

## How generation works

Training teaches a neural network to produce 32×32 grayscale pixels from a
random input and a class label. The exported PNG stores the trained weights.
In the browser, JavaScript runs the network's forward pass and sends its output
directly to the canvas. It does not apply image filters, add details, or clean
up the result. The grayscale values are copied to the canvas RGB channels.

This example uses a fully connected network trained with pixel-wise binary
cross-entropy. It is an actual neural image generator, but it is not a GAN and
does not promise photorealistic results.

## Notes

- Output size: 32×32 grayscale
- Classes: 0 = airplane, 1 = automobile, 2 = bird, 3 = cat, 4 = deer,
  5 = dog, 6 = frog, 7 = horse, 8 = ship, 9 = truck
- Same TensorRGBA format and pure JS decoder as the digit example
