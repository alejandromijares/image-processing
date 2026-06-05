# Module image-processing

Image-processing camera components for Viam.

## Models

- [`brad-grigsby:image-processing:color-correction`](brad-grigsby_image-processing_color-correction.md) — wraps a source camera and applies a 3×3 Color Correction Matrix fitted from a ColorChecker Classic, so colors stay consistent under your lighting. Corrects both the streaming `get_images` path and full-resolution stills captured via `DoCommand` (e.g. from the `brad-grigsby:canon:camera` module).
