# Examples

Additional models using the TensorRGBA format.

| Example | Description | Status |
|---------|-------------|--------|
| `mnist/` | Handwritten digit generator with sample float32 and int8 PNG models | Ready |
| `cifar10/` | Conditional RGB GAN for CIFAR-10 airplanes and ships | Train to export |

Each example is self-contained:

```
examples/<name>/
  train/
  web/
  README.md
```

The core format (header + float32/int8 packing + pure JS PNG decoder) is shared.
