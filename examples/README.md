# Examples

Additional models using the TensorRGBA format.

| Example | Description | Status |
|---------|-------------|--------|
| `animals/` | Black-and-white cat / dog generator (CIFAR-10) | Ready |
| `digits/` | Handwritten digit generator | Main example (see `/train` and `/web`) |

Each example is self-contained:

```
examples/<name>/
  train/
  web/
  README.md
```

The core format (header + float32/int8 packing + pure JS PNG decoder) is shared.
