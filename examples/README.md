# Examples

Additional models using the TensorRGBA format.

| Example | Description | Status |
|---------|-------------|--------|
| `mnist/` | Handwritten digit generator with sample float32 and int8 PNG models | Ready |
| `cifar10/` | Grayscale generator for all 10 CIFAR-10 classes | Ready |

Each example is self-contained:

```
examples/<name>/
  train/
  web/
  README.md
```

The core format (header + float32/int8 packing + pure JS PNG decoder) is shared.
