"""
image_io.py
-----------
Shared, studio-grade still-image IO for the image-processing models. Split out
so every component (color-correction today, others later) decodes and exports
files the same way.

The pipeline is built for studio photography (think Capture One), so the guiding
rules are:

* **16-bit, linear, no data thrown away early.** A Canon CR3 holds 14-bit linear
  sensor data. We demosaic it to 16-bit *linear* RGB (rawpy/libraw) and keep it
  as float through all color math, only encoding the sRGB transfer curve and
  dropping to 8-bit at the final delivery export. Auto-brightness is disabled so
  exposure is faithful to the capture.
* **Non-destructive.** Nothing here ever writes back into a ``.cr3`` - that's a
  proprietary undemosaiced container and can't be "re-saved". The raw stays the
  archival master; adjustments are recorded in a JSON sidecar (by the caller)
  and rendered out as separate files.
* **sRGB working/output space.** The color-correction CCM is fit against sRGB
  ColorChecker references, so we demosaic into sRGB primaries and tag exports
  with an sRGB ICC profile. (True wide-gamut output - ProPhoto etc. - would need
  the calibration reworked against wide-gamut references; that's a later job.)

Writer split, because no single library does it all cleanly:
  * 16-bit TIFF  -> tifffile  (embeds the sRGB ICC profile; the master)
  * 8-bit JPEG / PNG -> PIL    (embeds the sRGB ICC profile)
  * 16-bit PNG   -> OpenCV     (no ICC embed; documented limitation)
"""

import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageCms

# --- optional, lazily-imported backends ------------------------------------
# Each is only needed for a subset of formats, so import lazily and raise a
# clean, actionable error at point of use - the same pattern ptp.py uses for
# gphoto2.
try:
    import rawpy  # type: ignore

    _RAWPY_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    rawpy = None  # type: ignore
    _RAWPY_IMPORT_ERROR = exc

try:
    import tifffile  # type: ignore

    _TIFFFILE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    tifffile = None  # type: ignore
    _TIFFFILE_IMPORT_ERROR = exc

try:
    import cv2  # type: ignore

    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc

# Sensor-mosaic formats that must be demosaiced (via rawpy) rather than opened
# as a finished RGB image. Mirrors the RAW extensions ptp.py lists.
RAW_EXTS = (".cr3", ".cr2", ".nef", ".arw", ".raf", ".dng", ".rw2", ".orf")

# Export format keys -> (file-name suffix, bit depth). 16-bit variants are
# tagged ``_16`` so they don't collide with their 8-bit siblings in one folder.
EXPORT_FORMATS: Dict[str, Dict[str, object]] = {
    "tiff16": {"suffix": "_16.tif", "bits": 16},
    "tiff8":  {"suffix": ".tif",    "bits": 8},
    "jpeg":   {"suffix": ".jpg",    "bits": 8},
    "png16":  {"suffix": "_16.png", "bits": 16},
    "png8":   {"suffix": ".png",    "bits": 8},
}

_SRGB_ICC: Optional[bytes] = None


def _srgb_icc_bytes() -> bytes:
    """Cached sRGB ICC profile, generated once via littleCMS (bundled in PIL)."""
    global _SRGB_ICC
    if _SRGB_ICC is None:
        _SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _SRGB_ICC


# ---------------------------------------------------------------------------
# sRGB transfer functions (canonical home; color_correction imports these)
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB inverse gamma: gamma-encoded [0,1] -> linear light [0,1]."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """sRGB gamma: linear light [0,1] -> gamma-encoded [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def is_raw(path: str) -> bool:
    return path.lower().endswith(RAW_EXTS)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _rawpy_wb_kwargs(white_balance: Union[str, Sequence[float], None]) -> dict:
    """Translate a white-balance option into rawpy.postprocess kwargs."""
    if white_balance is None or white_balance in ("none", "daylight"):
        return {}  # libraw's fixed daylight WB
    if isinstance(white_balance, str):
        wb = white_balance.lower()
        if wb in ("camera", "as-shot", "as_shot"):
            return {"use_camera_wb": True}
        if wb == "auto":
            return {"use_auto_wb": True}
        raise ValueError(
            f"unknown white_balance {white_balance!r}; use 'camera', 'auto', "
            f"'daylight', or 4 raw multipliers [r, g, b, g2]"
        )
    mults = [float(v) for v in white_balance]
    if len(mults) != 4:
        raise ValueError("white_balance multipliers must be 4 values [r, g, b, g2]")
    return {"user_wb": mults}


def load_linear_rgb(
    path: str,
    *,
    white_balance: Union[str, Sequence[float], None] = "camera",
    exposure_stops: float = 0.0,
) -> np.ndarray:
    """
    Load an image file into a **linear-light** float32 RGB array in [0, 1],
    sRGB primaries.

    RAW files are demosaiced with rawpy at 16-bit, linear gamma, auto-brightness
    off, into the sRGB color space - so the only thing left to apply downstream
    is the CCM (in linear) and the output transfer curve. ``white_balance`` and
    ``exposure_stops`` are applied here, at the raw stage, where they belong.

    Non-RAW inputs (JPEG/PNG/TIFF) are assumed to be sRGB-encoded; they're read
    via PIL and linearized. They can't carry more than their stored precision.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"image file not found at {path!r}; if this came from the PTP "
            f"camera, configure its `download_dir` so captures are persisted to "
            f"disk, and make sure both components run on the same machine"
        )

    if is_raw(path):
        if rawpy is None:
            raise RuntimeError(
                f"cannot decode RAW file {os.path.basename(path)!r}: rawpy is "
                f"not available ({_RAWPY_IMPORT_ERROR}); install `rawpy` "
                f"(which bundles libraw) to enable RAW (CR3/NEF/ARW/...) support"
            )
        kwargs = dict(
            output_bps=16,
            gamma=(1, 1),            # linear light; we encode the curve at export
            no_auto_bright=True,     # faithful exposure, no surprise stretch
            output_color=rawpy.ColorSpace.sRGB,
        )
        kwargs.update(_rawpy_wb_kwargs(white_balance))
        if exposure_stops:
            # exp_shift is a linear multiplier; libraw clamps it to [0.25, 8].
            kwargs["exp_correc"] = True
            kwargs["exp_shift"] = float(np.clip(2.0 ** exposure_stops, 0.25, 8.0))
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(**kwargs)
        return (np.asarray(rgb16, dtype=np.float32) / 65535.0)

    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return srgb_to_linear(arr).astype(np.float32)


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------

def _encode_srgb(linear_rgb: np.ndarray, bits: int) -> np.ndarray:
    """Linear float RGB -> sRGB-gamma integer array at the given bit depth."""
    srgb = linear_to_srgb(linear_rgb)
    if bits == 16:
        return np.rint(srgb * 65535.0).clip(0, 65535).astype(np.uint16)
    return np.rint(srgb * 255.0).clip(0, 255).astype(np.uint8)


def _write_one(linear_rgb: np.ndarray, dest: str, fmt: str, quality: int) -> str:
    spec = EXPORT_FORMATS[fmt]
    bits = int(spec["bits"])
    data = _encode_srgb(linear_rgb, bits)
    icc = _srgb_icc_bytes()

    if fmt == "tiff16":
        if tifffile is None:
            raise RuntimeError(
                f"cannot write 16-bit TIFF: tifffile is not available "
                f"({_TIFFFILE_IMPORT_ERROR}); install `tifffile`"
            )
        # ICC profile lives in TIFF tag 34675 (UNDEFINED bytes). Best-effort:
        # if a tifffile version rejects the extratag form, fall back to no ICC.
        try:
            tifffile.imwrite(
                dest, data, photometric="rgb", compression="adobe_deflate",
                extratags=[(34675, 7, len(icc), icc, True)],
            )
        except Exception:
            tifffile.imwrite(dest, data, photometric="rgb", compression="adobe_deflate")
    elif fmt == "tiff8":
        Image.fromarray(data).save(dest, format="TIFF", icc_profile=icc)
    elif fmt == "jpeg":
        Image.fromarray(data).save(dest, format="JPEG", quality=quality, icc_profile=icc)
    elif fmt == "png8":
        Image.fromarray(data).save(dest, format="PNG", icc_profile=icc)
    elif fmt == "png16":
        if cv2 is None:
            raise RuntimeError(
                f"cannot write 16-bit PNG: OpenCV is not available "
                f"({_CV2_IMPORT_ERROR}); install `opencv-python-headless`"
            )
        # cv2 wants BGR; it does not embed an ICC profile (consumers assume sRGB).
        cv2.imwrite(dest, data[..., ::-1])
    else:  # pragma: no cover - guarded by EXPORT_FORMATS membership
        raise ValueError(f"unknown export format {fmt!r}")
    return dest


def export_renditions(
    linear_rgb: np.ndarray,
    out_dir: str,
    stem: str,
    formats: Sequence[str],
    *,
    quality: int = 95,
) -> Dict[str, str]:
    """
    Write ``linear_rgb`` (linear float RGB) to ``out_dir/<stem><suffix>`` for
    each requested format. Returns {format_key: written_path}.
    """
    unknown = [f for f in formats if f not in EXPORT_FORMATS]
    if unknown:
        raise ValueError(
            f"unknown export format(s) {unknown}; valid: {sorted(EXPORT_FORMATS)}"
        )
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for fmt in formats:
        dest = os.path.join(out_dir, stem + str(EXPORT_FORMATS[fmt]["suffix"]))
        written[fmt] = _write_one(linear_rgb, dest, fmt, quality)
    return written


def linear_to_jpeg_base64(linear_rgb: np.ndarray, max_dim: int = 1024, quality: int = 90) -> str:
    """
    Small sRGB JPEG preview (base64) for the control tab / DoCommand response -
    downsized so we never push a full-res still back over gRPC.
    """
    import base64
    from io import BytesIO

    rgb8 = _encode_srgb(linear_rgb, 8)
    img = Image.fromarray(rgb8)
    img.thumbnail((max_dim, max_dim))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
