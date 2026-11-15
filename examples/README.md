# Examples

Models using the TensorRGBA format.

| Example | Description | Status |
|---------|-------------|--------|
| `mnist/` | Handwritten digit generator | Ready |
| `cifar10/` | Conditional RGB GAN (airplane / ship) | Ready |
| `classifier/` | MNIST digit classifier (draw → predict) | Ready |

Each example is self-contained:

```
examples/<name>/
  train/
  web/
  README.md
```

The core format (header + float32/int8 packing + pure JS PNG decoder) is shared.
