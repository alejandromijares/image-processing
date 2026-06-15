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
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageCms
from viam.logging import getLogger

LOGGER = getLogger(__name__)


@contextmanager
def log_duration(logger, label: str):
    """Debug-log how long the enclosed block took, tagged `[timing]` so the
    pipeline's step durations are easy to grep out of a debug log."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.debug(f"[timing] {label}: {time.perf_counter() - start:.2f}s")

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

# Demosaic algorithms we expose, by rawpy.DemosaicAlgorithm name. RAW is soft
# before demosaic; the algorithm choice trades sharpness against artifacts.
# DHT is a high-quality default (sharper than libraw's stock AHD with few maze
# artifacts). AMAZE/LMMSE are deliberately omitted: they need the GPL2/GPL3
# demosaic packs that the bundled libraw wheels don't ship.
DEMOSAIC_ALGORITHMS = ("DHT", "AHD", "AAHD", "DCB", "VNG", "PPG")
DEFAULT_DEMOSAIC = "DHT"

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

def _demosaic_algorithm(name: str):
    """Resolve a demosaic name to a ``rawpy.DemosaicAlgorithm``, with a clean
    error if the bundled libraw can't do it (e.g. the GPL-only AMAZE/LMMSE)."""
    if name not in DEMOSAIC_ALGORITHMS:
        raise ValueError(
            f"unknown demosaic {name!r}; valid: {', '.join(DEMOSAIC_ALGORITHMS)}"
        )
    algo = getattr(rawpy.DemosaicAlgorithm, name)
    if not getattr(algo, "isSupported", True):
        raise RuntimeError(
            f"demosaic {name!r} isn't available in this libraw build; "
            f"choose another of {', '.join(DEMOSAIC_ALGORITHMS)}"
        )
    return algo


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
    user_flip: Optional[int] = None,
    half_size: bool = False,
    demosaic: str = DEFAULT_DEMOSAIC,
) -> np.ndarray:
    """
    Load an image file into a **linear-light** float32 RGB array in [0, 1],
    sRGB primaries.

    RAW files are demosaiced with rawpy at 16-bit, linear gamma, auto-brightness
    off, into the sRGB color space - so the only thing left to apply downstream
    is the CCM (in linear) and the output transfer curve. ``white_balance`` and
    ``exposure_stops`` are applied here, at the raw stage, where they belong.

    ``user_flip`` overrides libraw's orientation handling (default ``None`` =
    auto-rotate from EXIF). Pass ``0`` to disable rotation so the output lines
    up pixel-for-pixel with ``raw_image_visible`` / ``render_raw_for_detection``
    - used during color calibration so detected patch centres map across renders.

    ``half_size`` demosaics RAW at half resolution (each 2x2 CFA quad becomes
    one pixel - roughly 4x faster). Only for preview/throwaway renders; deliver
    exports from a full-size decode. Ignored for non-RAW inputs.

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
        if not half_size:
            # half_size bins each 2x2 CFA quad, so there's nothing to demosaic.
            kwargs["demosaic_algorithm"] = _demosaic_algorithm(demosaic)
        if exposure_stops:
            # exp_shift is a linear multiplier (rawpy enables the exposure
            # correction automatically when it's set); libraw clamps it to [0.25, 8].
            kwargs["exp_shift"] = float(np.clip(2.0 ** exposure_stops, 0.25, 8.0))
        if user_flip is not None:
            kwargs["user_flip"] = int(user_flip)
        if half_size:
            kwargs["half_size"] = True
        name = os.path.basename(path)
        size_mb = os.path.getsize(path) / 1e6
        label = "16-bit linear (half size)" if half_size else "16-bit linear"
        with log_duration(LOGGER, f"demosaic {name} ({size_mb:.1f} MB) to {label}"):
            with rawpy.imread(path) as raw:
                rgb16 = raw.postprocess(**kwargs)
        return (np.asarray(rgb16, dtype=np.float32) / 65535.0)

    with log_duration(LOGGER, f"decode + linearize {os.path.basename(path)}"):
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        return srgb_to_linear(arr).astype(np.float32)


# ---------------------------------------------------------------------------
# RAW colour-calibration helpers (ColorChecker white balance + detection)
# ---------------------------------------------------------------------------

def render_raw_for_detection(path: str) -> np.ndarray:
    """
    Demosaic a RAW into an 8-bit sRGB ``(H, W, 3)`` RGB array for ColorChecker
    *detection*.

    Rendered with camera white balance and a display gamma so the chart looks
    natural to the detector, and crucially with ``user_flip=0`` so the result is
    in the same orientation and (near-identical) resolution as
    ``raw_image_visible`` - letting patch centres found here map straight onto
    the raw CFA for white-balance sampling, and onto a matching linear render
    (``load_linear_rgb(..., user_flip=0)``) for the CCM fit.
    """
    if not is_raw(path):
        raise ValueError(f"render_raw_for_detection expects a RAW file, got {path!r}")
    if rawpy is None:
        raise RuntimeError(
            f"cannot decode RAW file {os.path.basename(path)!r}: rawpy is not "
            f"available ({_RAWPY_IMPORT_ERROR}); install `rawpy`"
        )
    with rawpy.imread(path) as raw:
        rgb8 = raw.postprocess(
            output_bps=8,
            no_auto_bright=True,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            user_flip=0,
        )
    return np.asarray(rgb8, dtype=np.uint8)


def compute_raw_wb_multipliers(
    path: str,
    boxes_norm: Sequence[Sequence[float]],
    *,
    saturation_fraction: float = 0.95,
) -> List[float]:
    """
    Measure ``[r, g, b, g2]`` white-balance multipliers (the format rawpy's
    ``user_wb`` wants) from the raw Bayer/CFA samples under neutral patches.

    This is the correct place to compute white balance: ``user_wb`` is applied
    to the CFA channels *before* demosaic, so we read the raw sensor values - not
    the demosaiced RGB - average each of the four CFA channels (R, G1, B, G2)
    over the neutral regions after subtracting the per-channel black level, and
    return multipliers that drive a neutral patch to equal channel values
    (greens normalised to ~1.0).

    Parameters
    ----------
    path : a RAW file path.
    boxes_norm : neutral-patch regions as ``(x0, y0, x1, y1)`` boxes, each
        coordinate a fraction in [0, 1] of the detection render's width/height
        (which matches ``raw_image_visible`` since both use ``user_flip=0``).
        Use mid-grey patches (Neutral 8 / 6.5) - not the white patch (clips) or
        black (noisy).
    saturation_fraction : skip CFA samples at or above this fraction of the
        white level, so a clipped highlight can't skew the average.
    """
    if rawpy is None:
        raise RuntimeError(
            f"cannot decode RAW file {os.path.basename(path)!r}: rawpy is not "
            f"available ({_RAWPY_IMPORT_ERROR}); install `rawpy`"
        )
    if not boxes_norm:
        raise ValueError("compute_raw_wb_multipliers needs at least one neutral region")

    with rawpy.imread(path) as raw:
        cfa = raw.raw_image_visible.astype(np.float32)
        colors = np.asarray(raw.raw_colors_visible)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)
        height, width = cfa.shape

        region = np.zeros((height, width), dtype=bool)
        for box in boxes_norm:
            x0, y0, x1, y1 = box
            ix0 = max(0, min(width - 1, int(round(x0 * width))))
            ix1 = max(0, min(width, int(round(x1 * width))))
            iy0 = max(0, min(height - 1, int(round(y0 * height))))
            iy1 = max(0, min(height, int(round(y1 * height))))
            if ix1 > ix0 and iy1 > iy0:
                region[iy0:iy1, ix0:ix1] = True
        if not region.any():
            raise ValueError(
                "neutral-patch regions mapped to an empty area of the raw frame"
            )

        sat_level = white * float(saturation_fraction)
        # libraw colour indices: 0=R, 1=G1, 2=B, 3=G2 (second green).
        avg = np.full(4, np.nan, dtype=np.float64)
        for c in range(4):
            sel = region & (colors == c) & (cfa < sat_level)
            if sel.any():
                bl = float(black[c]) if c < black.size else float(black[0])
                vals = cfa[sel] - bl
                vals = vals[vals > 0]
                if vals.size:
                    avg[c] = float(vals.mean())

    r, g1, b, g2 = avg
    # Sensors that label both greens as one colour have no index-3 samples.
    if np.isnan(g2):
        g2 = g1
    if np.isnan(g1):
        g1 = g2
    greens = [v for v in (g1, g2) if not np.isnan(v)]
    if not greens or np.isnan(r) or np.isnan(b) or r <= 0 or b <= 0:
        raise ValueError(
            "could not measure all CFA channels under the neutral patch; the "
            "chart may be over- or under-exposed, or the region missed the patch"
        )
    g_ref = float(np.mean(greens))
    wb = [g_ref / r, g_ref / g1, g_ref / b, g_ref / g2]
    if not all(np.isfinite(m) for m in wb) or any(m <= 0 for m in wb):
        raise ValueError(f"computed non-physical white-balance multipliers {wb}")
    return [float(m) for m in wb]


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delivery tone curves (the "look", layered on top of the accurate render)
# ---------------------------------------------------------------------------
# The CCM + exposure produce colour-accurate, colorimetric output: a mid-grey
# card lands on its true sRGB value (~160). Developers like Capture One instead
# apply a default tone curve that lifts the midtones well above that for a
# brighter, punchier delivery look. These optional curves reproduce that as a
# choice on top of the accurate render - the CCM (hue) is untouched; only
# lightness/contrast changes. Anchors are sRGB 0-255 (accurate input -> look
# output), read off the ColorChecker neutral row (our accurate export vs a
# Capture One export of the same frame): "bright" matches Capture One, "medium"
# is roughly half that lift. Endpoints are pinned so black stays black, white
# white. ``none`` (the default) is the identity - pure colorimetric output.
TONE_OPTIONS: Tuple[str, ...] = ("none", "medium", "bright")
_TONE_CURVES: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {
    "bright": ([0, 52, 85, 122, 160, 200, 243, 255],
               [0, 48, 105, 160, 200, 224, 240, 255]),
    "medium": ([0, 52, 85, 122, 160, 200, 243, 255],
               [0, 50,  95, 141, 180, 212, 242, 255]),
}
_TONE_LUT_SIZE = 4096
_TONE_LUT_CACHE: Dict[str, np.ndarray] = {}


def _monotone_cubic_lut(x: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite through (x, y) (both in [0,1]),
    sampled into a ``size``-entry LUT over [0,1]. Monotone => no tonal overshoot
    or inversion; smooth => none of the piecewise-linear 'mach band' kinks a
    straight interpolation would leave in a gradient (e.g. a seamless backdrop)."""
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.empty_like(y)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(delta.size):
        if delta[i] == 0.0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            t = a * a + b * b
            if t > 9.0:
                tau = 3.0 / np.sqrt(t)
                m[i], m[i + 1] = tau * a * delta[i], tau * b * delta[i]
    grid = np.linspace(0.0, 1.0, size)
    idx = np.clip(np.searchsorted(x, grid) - 1, 0, x.size - 2)
    dx = x[idx + 1] - x[idx]
    t = (grid - x[idx]) / dx
    t2, t3 = t * t, t * t * t
    lut = (
        (2 * t3 - 3 * t2 + 1) * y[idx]
        + (t3 - 2 * t2 + t) * dx * m[idx]
        + (-2 * t3 + 3 * t2) * y[idx + 1]
        + (t3 - t2) * dx * m[idx + 1]
    )
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def _tone_lut(tone: str) -> np.ndarray:
    if tone not in _TONE_LUT_CACHE:
        xs, ys = _TONE_CURVES[tone]
        _TONE_LUT_CACHE[tone] = _monotone_cubic_lut(
            np.asarray(xs, dtype=np.float64) / 255.0,
            np.asarray(ys, dtype=np.float64) / 255.0,
            _TONE_LUT_SIZE,
        )
    return _TONE_LUT_CACHE[tone]


def apply_tone_curve(srgb01: np.ndarray, tone: Optional[str]) -> np.ndarray:
    """Map gamma-encoded sRGB in [0,1] through a delivery tone curve.
    ``None``/``"none"`` is the identity (colour-accurate, colorimetric output)."""
    if not tone or tone == "none":
        return srgb01
    if tone not in _TONE_CURVES:
        raise ValueError(f"unknown tone {tone!r}; valid: {', '.join(TONE_OPTIONS)}")
    lut = _tone_lut(tone)
    grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float32)
    return np.interp(np.clip(srgb01, 0.0, 1.0), grid, lut).astype(np.float32)


# ---------------------------------------------------------------------------
# Capture sharpening
# ---------------------------------------------------------------------------
# RAW is soft before sharpening - every raw developer (Capture One, Lightroom)
# applies a default capture sharpen, which is why an unsharpened export looks
# blurry next to theirs. This is a luminance-only unsharp mask: the high-pass is
# computed on luma and added back to all channels, so edges crisp up without the
# colour fringing per-channel sharpening causes. Presets are (amount, sigma_px);
# sigma is at full export resolution. ``none`` (default) is off.
SHARPEN_OPTIONS: Tuple[str, ...] = ("none", "light", "medium", "strong")
_SHARPEN_PRESETS: Dict[str, Tuple[float, float]] = {
    "light":  (0.5, 0.8),
    "medium": (1.0, 0.9),
    "strong": (1.6, 1.1),
}
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def apply_sharpen(srgb01: np.ndarray, sharpen: Optional[str]) -> np.ndarray:
    """Luminance unsharp-mask on gamma-encoded sRGB [0,1]. ``None``/``"none"``
    is a no-op."""
    if not sharpen or sharpen == "none":
        return srgb01
    if sharpen not in _SHARPEN_PRESETS:
        raise ValueError(
            f"unknown sharpen {sharpen!r}; valid: {', '.join(SHARPEN_OPTIONS)}"
        )
    if cv2 is None:
        raise RuntimeError(
            f"capture sharpening needs OpenCV, which isn't available "
            f"({_CV2_IMPORT_ERROR}); install `opencv-python-headless`"
        )
    amount, sigma = _SHARPEN_PRESETS[sharpen]
    luma = srgb01 @ _LUMA
    blurred = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = (luma - blurred)[..., None]
    return np.clip(srgb01 + amount * detail, 0.0, 1.0).astype(np.float32)


def _encode_srgb(
    linear_rgb: np.ndarray, bits: int,
    tone: Optional[str] = None, sharpen: Optional[str] = None,
) -> np.ndarray:
    """Linear float RGB -> sRGB-gamma integer array at the given bit depth,
    optionally through a delivery ``tone`` curve and a capture ``sharpen`` pass
    (both applied in sRGB space)."""
    srgb = apply_sharpen(apply_tone_curve(linear_to_srgb(linear_rgb), tone), sharpen)
    if bits == 16:
        return np.rint(srgb * 65535.0).clip(0, 65535).astype(np.uint16)
    return np.rint(srgb * 255.0).clip(0, 255).astype(np.uint8)


def _write_one(
    linear_rgb: np.ndarray, dest: str, fmt: str, quality: int,
    tone: Optional[str] = None, sharpen: Optional[str] = None,
) -> str:
    spec = EXPORT_FORMATS[fmt]
    bits = int(spec["bits"])
    data = _encode_srgb(linear_rgb, bits, tone, sharpen)
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
    tone: Optional[str] = None,
    sharpen: Optional[str] = None,
) -> Dict[str, str]:
    """
    Write ``linear_rgb`` (linear float RGB) to ``out_dir/<stem><suffix>`` for
    each requested format. Returns {format_key: written_path}. ``tone`` applies
    an optional delivery tone curve (see ``apply_tone_curve``) and ``sharpen`` a
    capture unsharp mask (see ``apply_sharpen``) to every export.
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
        with log_duration(LOGGER, f"export {fmt} -> {os.path.basename(dest)}"):
            written[fmt] = _write_one(linear_rgb, dest, fmt, quality, tone, sharpen)
    return written


def linear_to_jpeg_base64(
    linear_rgb: np.ndarray, max_dim: int = 1024, quality: int = 90,
    tone: Optional[str] = None, sharpen: Optional[str] = None,
) -> str:
    """
    Small sRGB JPEG preview (base64) for the control tab / DoCommand response -
    downsized so we never push a full-res still back over gRPC.
    """
    import base64
    from io import BytesIO

    with log_duration(LOGGER, "encode base64 JPEG preview"):
        # The gamma encode is per-pixel float math - on a full-res still it
        # dwarfs everything else here. Stride-sample the linear array down to
        # ~2x the target first (cheap view, no copy of the full frame), and let
        # thumbnail()'s proper filter do the final clean resize.
        h, w = linear_rgb.shape[:2]
        stride = max(1, max(h, w) // (2 * max_dim))
        if stride > 1:
            linear_rgb = linear_rgb[::stride, ::stride]
        rgb8 = _encode_srgb(linear_rgb, 8, tone, sharpen)
        img = Image.fromarray(rgb8)
        img.thumbnail((max_dim, max_dim))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
