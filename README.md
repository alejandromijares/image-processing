# Module image-processing

Image-processing camera components for Viam.

## Models

- [`brad-grigsby:image-processing:color-correction`](brad-grigsby_image-processing_color-correction.md) — wraps a source camera and applies a 3×3 Color Correction Matrix fitted from a ColorChecker Classic, so colors stay consistent under your lighting. Corrects both the streaming `get_images` path and full-resolution stills captured via `DoCommand` (e.g. from the `brad-grigsby:canon:camera` module).
- [`brad-grigsby:image-processing:ptp`](brad-grigsby_image-processing_ptp.md) — talks directly to a USB-connected still camera (Canon/Nikon/Sony/etc.) over PTP via libgphoto2. Capture stills, list the card, and download images over a USB-C cable through `DoCommand`; `get_images` streams a live-view preview.
