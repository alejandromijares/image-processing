# Model brad-grigsby:image-processing:ptp

A camera component that talks **directly** to a still camera over USB (USB-C)
using **PTP** (Picture Transfer Protocol), via
[libgphoto2](http://www.gphoto.org/). Use it to pull images off a Canon / Nikon
/ Sony / Fujifilm / etc. body without an SD-card reader.

Unlike `color-correction`, this model does not wrap another Viam camera — it
owns the USB connection itself. libgphoto2 only lets one process hold the camera
at a time, so all access is serialized internally.

It works two ways:

- **Streaming** — `get_images` returns a live-view preview frame (a downsized
  JPEG from the camera's live view), so the control tab shows what the camera
  sees. If your body has no live view, it falls back to the newest still on the
  card.
- **DoCommand** — the real PTP workflow: trip the shutter, list the card,
  download files, and delete them.

## Requirements

- A **data-capable** USB-C cable (charge-only cables won't enumerate the camera).
- The camera powered on, and **no other app holding it** (close Photos / Image
  Capture on macOS, EOS Utility, or any running `gphoto2`).
- `gphoto2` Python package (declared in `requirements.txt`; it bundles
  libgphoto2 on Linux and macOS). Not supported on Windows.

## Configuration

```json
{
  "download_dir": "/home/pi/captures",
  "delete_after_download": false
}
```

All attributes are optional — with an empty config it binds to the first camera
detected on USB.

### Attributes

| Name                    | Type    | Inclusion | Description                                                                                   |
|-------------------------|---------|-----------|-----------------------------------------------------------------------------------------------|
| `camera_model`          | string  | Optional  | Substring to match a specific body when several are connected (e.g. `"Canon"`, `"R5"`).       |
| `port`                  | string  | Optional  | Exact libgphoto2 port to bind (e.g. `"usb:001,005"`). Use `summary` to discover it.           |
| `download_dir`          | string  | Optional  | Local directory to also save downloaded images to. Created if missing. Required for `download_all`. |
| `delete_after_download` | boolean | Optional  | Delete each file from the camera after a successful download. Default `false`.                |

## DoCommand

### Capture a still

Trips the shutter, downloads the resulting full-resolution still, and returns it
as base64.

```json
{ "capture": {} }
```

Returns `{"name", "path", "mime_type", "image_base64", "saved_to", "size"}`
(`saved_to` is `null` unless `download_dir` is set).

### List files

```json
{ "list_files": { "new_only": true } }
```

Returns `{"files": ["/store_00010001/DCIM/100CANON/IMG_0042.JPG", ...], "count": N}`.
`new_only` (optional) lists only files not yet downloaded this session.

### Download a file

```json
{ "download": { "path": "/store_00010001/DCIM/100CANON/IMG_0042.JPG" } }
```
```json
{ "download": { "latest": true } }
```

Returns the same shape as `capture`. RAW files (`.cr3`, `.nef`, `.arw`, …)
download fine but come back with `mime_type: "application/octet-stream"`.

### Download everything to disk

```json
{ "download_all": { "new_only": true } }
```

Requires `download_dir` (we don't base64 a whole card back over gRPC). Saves
each image to `download_dir` and returns `{"saved": [...paths], "count": N}`.

### Delete a file

```json
{ "delete": { "path": "/store_00010001/DCIM/100CANON/IMG_0042.JPG" } }
```

### Camera summary

```json
{ "summary": {} }
```

Returns the camera `model`, USB `port`, and libgphoto2's capability `summary`
text. Handy for finding the `port` value to pin in config.

## Typical workflow

1. Plug the camera into the machine with a data USB-C cable and power it on.
2. Configure the component (optionally set `download_dir`).
3. Run `summary` to confirm it's detected.
4. Use `capture` to shoot-and-pull, or `list_files` + `download` to retrieve
   images already on the card. Use `download_all` to sync the whole card to
   disk.

## Troubleshooting

- **"no PTP camera detected on USB"** — check the cable is data-capable, the
  camera is on, and no other app holds it. On Linux you may need udev
  permissions (add your user to the `plugdev` group) so viam-server can open the
  device.
- **"live-view preview unavailable"** — your body doesn't expose live view over
  PTP; `get_images` falls back to the newest still automatically.
