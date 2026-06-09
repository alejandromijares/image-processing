# Model brad-grigsby:image-processing:color-correction

A camera component that wraps another (source) camera and applies a 3×3 Color
Correction Matrix (CCM) to its images. The CCM is fitted from a photo of a
Calibrite / X-Rite **ColorChecker Classic** (24 patches), so colors come out
consistent under your lighting.

It works two ways:

- **Streaming** — `get_images` proxies the source camera and color-corrects
  every JPEG/PNG frame (names preserved). Used by the control tab, data
  manager, and vision services.
- **DoCommand** — a studio-grade RAW developer. `capture` triggers a still on
  the source camera (e.g. the `ptp` model), and `develop` processes RAW/image
  files already on disk. Both run the same pipeline:

  - RAW (CR3/NEF/ARW/RAF/DNG/…) is demosaiced to **16-bit linear** with
    [rawpy](https://pypi.org/project/rawpy/) (auto-brightness off, white balance
    applied at the raw stage), so no tonal precision is lost before correction.
  - White balance and the CCM are applied in **linear light**.
  - The result is written out as rendered exports (16-bit TIFF, JPEG, etc.),
    tagged with an sRGB ICC profile.
  - It's **non-destructive**: the original RAW is never modified. The
    adjustments are recorded in a `<name>.json` sidecar next to it, the way
    Capture One / Lightroom keep edits separate from the negative.

> **Color space:** output is sRGB, because the CCM is fitted against sRGB
> ColorChecker references. True wide-gamut output (ProPhoto, etc.) would need the
> calibration reworked against wide-gamut references.

## Requirements

The RAW pipeline needs native libraries (`rawpy` bundling LibRaw, `tifffile`,
`opencv-python-headless`). These install from `requirements.txt` as prebuilt
wheels on `linux/amd64`, `linux/arm64`, and `darwin/arm64`. See the
[module README](README.md#system-requirements) for the minimal/headless cases
where a system package (`libraw-dev`, `libglib2.0-0`) is needed — `setup.sh`
installs those automatically on Debian/Ubuntu.

## Configuration

```json
{
  "camera": "my-ptp-cam",
  "ccm": [
    [1.16, -0.08, -0.04],
    [-0.10, 1.28, -0.10],
    [0.00, 0.02, 0.72]
  ],
  "output_dir": "/photos/exports",
  "output_formats": ["tiff16", "jpeg"]
}
```

### Attributes

| Name             | Type         | Inclusion | Description                                                                             |
|------------------|--------------|-----------|-----------------------------------------------------------------------------------------|
| `camera`         | string       | Required  | Source camera to wrap; declared as a dependency. Required even for `develop`.           |
| `ccm`            | 3×3 array    | Optional  | Color correction matrix from `calibrate_color`. Omit to pass images through unchanged.  |
| `output_dir`     | string       | Optional  | Where `capture`/`develop` write exports. Default: next to the source file.              |
| `output_formats` | string[]     | Optional  | Any of `tiff16`/`tiff8`/`jpeg`/`png16`/`png8`. Default: all four.                       |
| `jpeg_quality`   | int          | Optional  | JPEG export quality. Default `95`.                                                      |
| `white_balance`  | string/array | Optional  | RAW white balance: `camera` (default), `auto`, `daylight`, or `[r,g,b,g2]` multipliers. |
| `write_sidecar`  | boolean      | Optional  | Write a `<name>.json` sidecar recording the development. Default `true`.                |

If no `ccm` is given, the component passes images through unchanged (identity
matrix); the RAW develop still runs (demosaic + export) but applies no color
correction. To get a matrix, run `calibrate_color` and copy the returned `ccm`
into this attribute.

### Export file naming

Exports are written to `output_dir` (or next to the source) as
`<stem><suffix>`: `tiff16` → `_16.tif`, `tiff8` → `.tif`, `jpeg` → `.jpg`,
`png16` → `_16.png`, `png8` → `.png`. When the **source itself** is a JPEG/PNG/
TIFF (not a RAW), exports get a `_corrected` suffix so the original is never
overwritten.

## DoCommand

### Calibrate

Place the ColorChecker Classic in frame, then fit a CCM. With `use_capture:
true` it calibrates from a full-resolution still via the source camera's
`capture` command; otherwise it uses the source's live/streaming frame. The
fitted matrix is applied immediately and returned — copy it into the `ccm`
config attribute to persist it across restarts.

```json
{
  "calibrate_color": {
    "use_capture": true
  }
}
```

Options: `use_capture` (bool), `capture_options` (object forwarded to the source
capture, default `{"af": true}`), `white_balance` (used when developing the RAW
for calibration), `patch_centers` (24 `[x, y]` pixel coords in ColorChecker
order — use this when the chart does not fill the frame), `radius` (int, patch
sampling radius).

Returns the fitted `ccm` (copy this into config) and a `delta_e` quality report
(mean/max color error before vs. after).

### Capture a corrected still

Triggers a full-resolution still on the source camera, develops it through the
pipeline, and writes the exports + sidecar. Returns a small base64 JPEG
**preview** (the full-resolution image stays on disk).

```json
{
  "capture": {
    "capture_options": { "af": true },
    "white_balance": "camera",
    "exposure_stops": 0,
    "output_formats": ["tiff16", "jpeg"]
  }
}
```

Options (all optional): `capture_options` (forwarded to the source's `capture`),
`white_balance`, `exposure_stops` (exposure compensation applied at the raw
stage), `output_formats`, `output_dir`. Each overrides the config default for
this call.

Returns:

```json
{
  "source_path": "/photos/IMG_0042.CR3",
  "exports": { "tiff16": "/photos/IMG_0042_16.tif", "jpeg": "/photos/IMG_0042.jpg" },
  "sidecar": "/photos/IMG_0042.json",
  "ccm_applied": true,
  "color_space": "sRGB",
  "image_base64": "<downsized JPEG preview>",
  "mime_type": "image/jpeg"
}
```

### Develop existing files (no camera)

Point at a RAW or image file already on disk — no camera trigger needed. Takes
the same options as `capture`.

```json
{ "develop": { "path": "/photos/IMG_0042.CR3" } }
```

Batch several files at once with `paths`:

```json
{ "develop": { "paths": ["/photos/a.CR3", "/photos/b.CR3"], "output_dir": "/exports" } }
```

A single `path` returns the same shape as `capture` (including a preview). A
`paths` list returns `{"developed": [ ...per-file results... ], "count": N}`
with previews omitted to keep the response small.

## Typical workflows

**Live capture (with the `ptp` model):**

1. Configure the `ptp` component with a `download_dir`, and configure
   `color-correction` with `camera` pointing at it. Both must run on the same
   machine (shared filesystem).
2. Frame the ColorChecker and run `calibrate_color` (`use_capture: true`) once;
   check the returned `delta_e.after.mean` is low (a few units).
3. Copy the returned `ccm` into the `ccm` config attribute and save.
4. Run `capture` to shoot, develop, and export in one step.

**Develop a folder of existing RAWs:**

1. Configure `ccm` (and `output_dir` if you don't want exports next to the
   originals).
2. Call `develop` with `paths` listing the CR3 files. Each gets its exports +
   sidecar; the RAWs are left untouched.
