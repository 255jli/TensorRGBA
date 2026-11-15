# Architecture notes

## MNIST digit generator

MLP:

```
noise (128) + one-hot (10) → 256 → 512 → 1024 → 784
```

Activation: ReLU + final Sigmoid.

Simple, fast, and easy to understand. Good first example.

## CIFAR-10 airplane and ship generator

Conditional DCGAN:

```
noise (128) + one-hot (2) → 2048 → 128×4×4 → 64×8×8 → 32×16×16 → RGB 32×32
```

Uses ConvTranspose2d and BatchNorm. The discriminator is used only during training. The PNG stores generator weights and BatchNorm statistics.

## MNIST classifier

MLP:

```
784 → 256 → 128 → 10
```

Shows the opposite direction of the generators: image → class probabilities. The browser demo lets you draw a digit and run the network.

## Format stability

The 64-byte header and the two packing modes (float32 / int8) are considered stable.  
New models should reuse the same header layout and the same pure-JS decoder.

Header byte 24 is used as an optional architecture id (0 = MLP generator, 1 = DCGAN, 2 = classifier).
