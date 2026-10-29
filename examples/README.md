# Examples

This folder will contain additional models packed with the TensorRGBA format.

Current planned examples:

- `digits/` — handwritten digit generator (the main example, currently lives in `/train` and `/web`)
- `animals/` — black-and-white cat / dog generator (future)
- `cleanup/` — simple watermark or background removal (future)
- `tiny-lm/` — extremely small character-level language model (experimental)

Each example should be self-contained:

```
examples/<name>/
  train/
  web/
  README.md
```

The core format (header + float32 / int8 packing + pure JS decoder) stays the same for every example.
