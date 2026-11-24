# Vantage logo

The **Surround** mark: the vehicle seen from above inside four coverage arcs, with the gaps between the arcs on the
car's axes (front, rear, left, right). It is the product in one glyph: a sensor rig and the ground it covers.

All files here are written by `packaging/make_brand.py` (regenerate rather than edit them by hand);
`packaging/make_icon.py` renders the application icons and the website's images from them.

| File | Use |
|---|---|
| `mark-white.svg`, `mark-black.svg` | the mark, 48 px and larger |
| `mark-32-*.svg` | the mark drawn for 24-40 px (heavier ring, wider gaps) |
| `mark-16-*.svg` | the mark drawn for 16-20 px |
| `wordmark-*.svg` | VANTAGE alone |
| `lockup-horizontal-*.svg` | mark + wordmark side by side: the default lockup |
| `lockup-stacked-*.svg` | mark over wordmark, for square spaces |
| `app-icon-*-dark.svg`, `app-icon-*-light.svg` | the icon tile (dark is the default) at each optical size |

## Rules

- **Colour:** one colour at a time: off-white `#EDEDE8` on near-black `#0A0B0B`, or near-black on `#F2F1EC` /
  white. No gradients, outlines, shadows or second colours inside the mark.
- **Clear space:** at least the ring's stroke × 4 (28 % of the mark's height) on every side.
- **Minimum size:** 16 px for the mark (use the 16 px drawing), 96 px wide for the horizontal lockup.
- **Lockup proportions:** the wordmark's capitals are 40 % of the mark's height, 40 % of it away from the mark.
  Don't rebuild a lockup from text: use the files.
- **Wordmark:** Archivo SemiBold (weight 600) at width 125, capitals tracked 0.30 em, converted to outlines. Archivo is
  licensed under the SIL Open Font License 1.1 (© 2020 The Archivo Project Authors); the font file is not in this
  repository.
- Don't rotate the mark: the car points up (forward).
