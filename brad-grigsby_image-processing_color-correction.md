# Model brad-grigsby:image-processing:color-correction

A camera component that wraps another (source) camera and applies a 3×3 Color
Correction Matrix (CCM) to its images. The CCM is fitted from a photo of a
Calibrite / X-Rite **ColorChecker Classic** (24 patches), so colors come out
consistent under your lighting.

It works two ways:

- **Streaming** — `get_images` proxies the source camera and color-corrects
  every JPEG/PNG frame (names preserved). Used by the control tab, data
  manager, and vision services.
- **DoCommand** — for cameras whose full-resolution stills are exposed through
  DoCommand (e.g. the `brad-grigsby:canon:camera` module), `capture` triggers a
  still on the source, corrects it, and returns it as base64 JPEG.

## Configuration

```json
{
  "camera": "my-canon",
  "ccm": [
    [1.16, -0.08, -0.04],
    [-0.10, 1.28, -0.10],
    [0.00, 0.02, 0.72]
  ]
}
```

### Attributes

| Name     | Type      | Inclusion | Description                                                  |
|----------|-----------|-----------|--------------------------------------------------------------|
| `camera` | string    | Required  | Name of the source camera to wrap. Declared as a dependency. |
| `ccm`    | 3×3 array | Optional  | The color correction matrix. Get one from `calibrate_color`. |

If no `ccm` is given, the component passes images through unchanged (identity
matrix). To get a matrix, run the `calibrate_color` DoCommand and copy the
returned `ccm` into this attribute.

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
capture, default `{"af": true}`), `patch_centers` (24 `[x, y]` pixel coords in
ColorChecker order — use this when the chart does not fill the frame), `radius`
(int, patch sampling radius).

Returns the fitted `ccm` (copy this into config) and a `delta_e` quality report
(mean/max color error before vs. after).

### Capture a corrected still

Triggers a full-resolution still on the source camera, color-corrects it, and
returns it as base64 JPEG. Options are forwarded to the source's `capture`.

```json
{
  "capture": { "af": true }
}
```

Returns `{"image_base64": "...", "mime_type": "image/jpeg", "path": "<source file path>"}`.

## Typical workflow

1. Configure with the source `camera` (no `ccm` yet).
2. Frame the ColorChecker and run `calibrate_color` once (check the returned
   `delta_e.after.mean` is low — a few units).
3. Copy the returned `ccm` matrix into the component's `ccm` config attribute
   and save the config.
4. Leave it running: `get_images` is corrected automatically, and `capture`
   returns corrected stills.
