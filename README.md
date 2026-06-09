# Module image-processing

Image-processing camera components for Viam.

## Models

- [`brad-grigsby:image-processing:color-correction`](brad-grigsby_image-processing_color-correction.md) — wraps a source camera and applies a 3×3 Color Correction Matrix fitted from a ColorChecker Classic, so colors stay consistent under your lighting. Corrects the streaming `get_images` path, and offers a non-destructive studio RAW developer via `DoCommand`: `capture` triggers and develops a still from the source camera (e.g. the `ptp` model), and `develop` processes existing CR3/NEF/… files already on disk. RAW is demosaiced to 16-bit linear, white-balanced and color-corrected, then exported (16-bit TIFF, JPEG, etc.) with the original left untouched.
- [`brad-grigsby:image-processing:ptp`](brad-grigsby_image-processing_ptp.md) — talks directly to a USB-connected still camera (Canon/Nikon/Sony/etc.) over PTP via libgphoto2. Capture stills, list the card, and download images over a USB-C cable through `DoCommand`; `get_images` streams a live-view preview.

## System requirements

`setup.sh` builds the Python venv on the target device. The RAW pipeline used by
`color-correction` depends on native libraries shipped inside their wheels
(`rawpy` bundles LibRaw, `opencv-python-headless` bundles its own libs,
`tifffile` is pure Python), so a normal install on `linux/amd64`, `linux/arm64`,
or `darwin/arm64` needs nothing extra.

On a minimal/headless target the wheels can still fail to load. `setup.sh`
detects this and, on Debian/Ubuntu, auto-installs the system packages:

- **`libraw-dev`** — if pip has to build `rawpy` from source (no matching wheel)
- **`libglib2.0-0`** — if OpenCV-headless can't load `libgthread-2.0.so`

On non-apt systems, install the LibRaw and glib equivalents for your OS
(e.g. `brew install libraw` on macOS) and re-run.
