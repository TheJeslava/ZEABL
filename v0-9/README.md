# v0-9 XLRS-lite Crop-Placement Ablation

v0-9 is copied from v0-8. The only inference change is the placement of the
high-resolution crop image token for questions containing a numeric reference
bounding box:

- v0-8 places the crop token after `<|im_start|>assistant`.
- v0-9 moves that same token immediately before
  `<|im_end|><|im_start|>assistant` so both the global image and crop belong to
  the user turn.

The prompt text, global-then-crop image order, global question coordinates,
model, processor, image resizing, generation settings, seed, attention,
extraction, scoring, and fallback behavior remain unchanged from v0-8. The
no-reference path is also unchanged.

The 150-sample experiment selects 50 evenly distributed rows from each numeric
reference-bbox category, using the same positions as the earlier v0-5 sample:

```text
Object properties/Object motion state   50 / 60
Object properties/Object color          50 / 800
Object properties/Object classification 50 / 800
```

Run it with:

```bash
./run_scripts/run_reference_50x3.sh
```

The authoritative output prefix is
`results/zoomearth-xlrs-reference-user-crop-150`.
