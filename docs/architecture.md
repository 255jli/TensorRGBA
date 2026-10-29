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

## Future convolutional upgrade

A DCGAN-style or PixelShuffle generator can be added later as a separate example.  
It will require a small convolution / transpose-convolution implementation on the JavaScript side.  
Until that is ready, the MLP remains the reference implementation.

## Format stability

The 64-byte header and the two packing modes (float32 / int8) are considered stable.  
New models should reuse the same header layout and the same pure-JS decoder.
