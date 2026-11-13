# Architecture notes

## Current digit generator

MLP:

```
noise (128) + one-hot (10) → 256 → 512 → 1024 → 784
```

Activation: ReLU + final Sigmoid.

Reasons for keeping MLP as the base example:

- Extremely simple pure-JS implementation
- Fast on low-end devices
- Easy to understand and modify
- Still produces recognizable digits after proper training

## CIFAR-10 airplane and ship generator

Conditional DCGAN:

```
noise (128) + one-hot (2) → 2048 → 128×4×4 → 64×8×8 → 32×16×16 → RGB 32×32
```

The discriminator is used only during training. The PNG stores the generator
weights and BatchNorm statistics; JavaScript performs its forward pass directly.

## Future higher-resolution upgrade

Higher resolutions can be added as a separate example with more upsampling
layers and matching JavaScript inference.

## Format stability

The 64-byte header and the two packing modes (float32 / int8) are considered stable.  
New models should reuse the same header layout and the same pure-JS decoder.
