"""
color_correction.py
--------------------
A Viam camera component that color-corrects images from a source camera using
a 3x3 Color Correction Matrix (CCM) fitted from a Calibrite / X-Rite
ColorChecker Classic.

Two ways to get corrected images out of this component:

1. Streaming path - ``get_images`` proxies the source camera, applies the CCM
   to every JPEG/PNG frame, and returns the corrected images (names preserved).
   This is what the control tab, data manager, and vision services use.

2. DoCommand path - the studio RAW workflow, for cameras (the PTP model, or
   the Canon CCAPI module) whose full-resolution stills are exposed through
   DoCommand rather than the streaming ``Images`` method:

       {"capture": {"white_balance": "camera",
                    "output_formats": ["tiff16", "jpeg"]}}
           -> trigger a still on the source camera; if it's a RAW (CR3/NEF/...)
              downloaded to disk, demosaic it to 16-bit *linear*, apply white
              balance + the CCM, and write rendered exports next to it - leaving
              the RAW untouched as the master. Returns the export paths, a JSON
              sidecar path recording the development, and a small base64 JPEG
              preview (the full image stays on disk).

   The source still arrives either inline as ``image_base64`` (small JPEGs from
   CCAPI) or as a downloaded file path in ``saved_to`` - the PTP RAW handoff.
   Wire the PTP component as this model's ``camera`` dependency and give PTP a
   ``download_dir`` so its captures land on disk where this model can read them.

       {"capture": {"defer": true}}
       {"capture_result": {"id": "<capture_id>", "wait_sec": 60}}
           -> pipelined capture for rigs that move between shots: ``capture``
              with ``defer`` returns {"capture_id", "status", "camera_path"}
              as soon as the shutter has fired (exposure done - the rig is
              free to move), while the USB download, half-size demosaic, CCM,
              and preview encode continue in the background.
              ``capture_result`` then returns {"source_path", "image_base64",
              ...} (or {"status": "pending"} if not done within ``wait_sec``).
              Deferred captures never write exports or a sidecar - run
              ``develop`` on the returned ``source_path`` when the files are
              actually needed. Requires the ptp model (its ``trigger``
              command) as the source camera.

   This is a non-destructive, Capture One-style pipeline: 16-bit linear math,
   no auto-brightness, the original RAW preserved, adjustments recorded in a
   sidecar. See image_io.py for the decode/export details and color-space notes.

   Relevant config attributes: ``output_dir`` (default: next to the source),
   ``output_formats`` (default ["tiff16", "jpeg", "png16", "png8"]),
   ``jpeg_quality`` (95), ``white_balance`` ("camera"), ``write_sidecar`` (true).

       {"develop": {"path": "/photos/IMG_0042.CR3"}}
       {"develop": {"paths": ["/photos/a.CR3", "/photos/b.CR3"]}}
           -> develop existing RAW/image file(s) already on disk through the
              same pipeline, with no camera trigger. Takes the same
              white_balance / exposure_stops / output_formats / output_dir
              options as ``capture``. A single ``path`` returns that file's
              result; ``paths`` returns {"developed": [...], "count": N}.

Calibration:

       {"calibrate_color": {}}
           -> grab a live-view frame, auto-detect the ColorChecker (cv2.mcc),
              fit a CCM, and return it. No white balance (no RAW from live view).
       {"calibrate_color": {"use_capture": true}}
           -> trigger a full-resolution RAW still, auto-detect the chart,
              measure white balance from the raw CFA under the neutral patches
              ([r,g,b,g2] for rawpy's user_wb), and fit the CCM under that same
              white balance. Returns both.
       {"calibrate_color": {"path": "/photos/chart.CR3"}}
           -> same, from a RAW already on disk (no camera trigger).

The chart is detected automatically anywhere in frame (cv2.mcc, from
opencv-contrib) - no clicking patch centres. Pass ``patch_centers`` (24 [x,y])
to override detection, or ``compute_wb: false`` to skip white balance.
`calibrate_color` returns the fitted 3x3 ``ccm`` and the 4-value
``white_balance``; copy them into the component's ``ccm`` / ``white_balance``
config attributes to make the calibration persist across restarts.
"""

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image
from typing_extensions import Self

from viam.app.data_client import DataClient
from viam.components.camera import Camera
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image
from viam.rpc.dial import Credentials, DialOptions, _dial_app
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from models.image_io import (
    EXPORT_FORMATS,
    compute_raw_wb_multipliers,
    export_renditions,
    is_raw,
    linear_to_jpeg_base64,
    linear_to_srgb,
    load_linear_rgb,
    render_raw_for_detection,
    srgb_to_linear,
)

# OpenCV's ColorChecker detector (cv2.mcc) lives in opencv-contrib; import lazily
# so the module still loads (and the streaming/develop paths work) on a host
# without it - calibration raises a clean, actionable error at point of use.
try:
    import cv2  # type: ignore

    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc

# Default delivery set when `output_formats` isn't configured. Override in
# config to trim it (e.g. just ["tiff16", "jpeg"] for a master + proof).
DEFAULT_OUTPUT_FORMATS = ["tiff16", "jpeg", "png16", "png8"]

# ---------------------------------------------------------------------------
# ColorChecker Classic reference values (24 patches, sRGB, D50 illuminant)
# Row order: dark skin -> white, left-to-right, top-to-bottom as the chart
# is oriented with the "colorchecker CLASSIC" text at the top.
# Source: Calibrite / X-Rite published sRGB reference (gamma-encoded, 0-255)
# ---------------------------------------------------------------------------
REFERENCE_SRGB = np.array([
    # Row 1
    [115,  82,  68],   # 1  Dark Skin
    [194, 150, 130],   # 2  Light Skin
    [ 98, 122, 157],   # 3  Blue Sky
    [ 87, 108,  67],   # 4  Foliage
    [133, 128, 177],   # 5  Blue Flower
    [103, 189, 170],   # 6  Bluish Green
    # Row 2
    [214, 126,  44],   # 7  Orange
    [ 80,  91, 166],   # 8  Purplish Blue
    [193,  90,  99],   # 9  Moderate Red
    [ 94,  60, 108],   # 10 Purple
    [157, 188,  64],   # 11 Yellow Green
    [224, 163,  46],   # 12 Orange Yellow
    # Row 3
    [ 56,  61, 150],   # 13 Blue
    [ 70, 148,  73],   # 14 Green
    [175,  54,  60],   # 15 Red
    [231, 199,  31],   # 16 Yellow
    [187,  86, 149],   # 17 Magenta
    [  8, 133, 161],   # 18 Cyan
    # Row 4 (neutral patches)
    [243, 243, 242],   # 19 White
    [200, 200, 200],   # 20 Neutral 8
    [160, 160, 160],   # 21 Neutral 6.5
    [122, 122, 121],   # 22 Neutral 5
    [ 85,  85,  85],   # 23 Neutral 3.5
    [ 52,  52,  52],   # 24 Black
], dtype=np.float32) / 255.0   # normalise to [0, 1]


# Canonical sRGB transfer functions live in image_io so the decode/export path
# and the color math agree exactly; aliased here to keep call sites readable.
_srgb_to_linear = srgb_to_linear
_linear_to_srgb = linear_to_srgb


def _fit_ccm(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Least-squares fit of a 3x3 Color Correction Matrix.

    Solves ``reference ~= measured @ CCM.T`` (each reference row = CCM @ measured_row).

    Parameters
    ----------
    measured  : (N, 3) float32, linear-light measured RGB, normalised [0, 1]
    reference : (N, 3) float32, linear-light reference RGB, normalised [0, 1]

    Returns
    -------
    ccm : (3, 3) float32
    """
    solution, _, _, _ = np.linalg.lstsq(measured, reference, rcond=None)
    return solution.T  # shape (3, 3)


# ---------------------------------------------------------------------------
# Automatic ColorChecker detection (cv2.mcc)
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 chart corners as [top-left, top-right, bottom-right, bottom-left]
    *of the image frame* (x+y / x-y heuristic).

    This fixes the winding only - it says nothing about which corner sits next
    to the dark-skin patch. The calibration render is deliberately unrotated
    (``user_flip=0``), so a portrait shot puts the chart on its side;
    ``_oriented_chart_grid`` below tries all four 90-degree assignments and
    keeps the one whose colors match the reference layout.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],  # top-left      (smallest x+y)
        pts[np.argmax(d)],  # top-right     (largest  x-y)
        pts[np.argmax(s)],  # bottom-right  (largest  x+y)
        pts[np.argmin(d)],  # bottom-left   (smallest x-y)
    ], dtype=np.float32)


def _orientation_score(measured_srgb: np.ndarray) -> float:
    """
    How well 24 sampled patch colors match REFERENCE_SRGB's layout: per-channel
    Pearson correlation, summed over R/G/B (max 3.0). Standardizing each channel
    makes the score invariant to exposure and to per-channel gain - so an
    uncorrected white-balance cast can't disguise the right orientation.
    Wrong rotations land near 0.
    """
    score = 0.0
    for c in range(3):
        m, r = measured_srgb[:, c], REFERENCE_SRGB[:, c]
        ms, rs = float(m.std()), float(r.std())
        if ms < 1e-6 or rs < 1e-6:
            continue
        score += float((((m - m.mean()) / ms) * ((r - r.mean()) / rs)).mean())
    return score


# Below this, no candidate rotation matched the reference layout - the detector
# most likely latched onto something that isn't a ColorChecker Classic.
_MIN_ORIENTATION_SCORE = 0.75


def _oriented_chart_grid(
    img_rgb: np.ndarray, box: np.ndarray, *, rows: int = 4, cols: int = 6
) -> Optional[Dict[str, Any]]:
    """
    Map a detected chart quad to the 24 patch centres in REFERENCE_SRGB order,
    robust to the chart sitting at any 90-degree rotation in the frame.

    Tries the four cyclic corner assignments, samples the patch colors each
    would imply, and keeps the orientation that correlates with the reference
    layout. Returns ``None`` if none does (false-positive detection).

    Returns a dict with ``centers`` (24, 2), ``neutral_boxes_norm`` (axis-
    aligned inner boxes over Neutral 8 / 6.5 for raw WB sampling),
    ``suggested_radius`` (patch-size-relative sampling half-width), and
    ``orientation_score``.
    """
    corners = _order_corners(box)
    h, w = img_rgb.shape[:2]
    img_f = img_rgb.astype(np.float32) / 255.0

    def make_grid(c0, c1, c2, c3):
        # c0->c1 spans the `cols` axis, c0->c3 the `rows` axis.
        def grid_point(u: float, v: float) -> np.ndarray:
            top = c0 + (c1 - c0) * u
            bot = c3 + (c2 - c3) * u
            return top + (bot - top) * v
        centers = np.zeros((rows * cols, 2), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                centers[r * cols + c] = grid_point((c + 0.5) / cols, (r + 0.5) / rows)
        return centers, grid_point

    def sample(centers: np.ndarray, radius: int) -> np.ndarray:
        out = np.zeros((len(centers), 3), dtype=np.float32)
        for i, (x, y) in enumerate(centers):
            xi, yi = int(round(float(x))), int(round(float(y)))
            x0, y0 = max(0, xi - radius), max(0, yi - radius)
            patch = img_f[y0:yi + radius, x0:xi + radius].reshape(-1, 3)
            if patch.size:
                out[i] = np.median(patch, axis=0)
        return out

    best_score, best = -np.inf, None
    cycle = list(corners)
    for _ in range(4):
        c0, c1, c2, c3 = cycle
        centers, grid_point = make_grid(c0, c1, c2, c3)
        patch_px = min(
            float(np.linalg.norm(c1 - c0)) / cols,
            float(np.linalg.norm(c3 - c0)) / rows,
        )
        radius = max(2, int(0.15 * patch_px))
        score = _orientation_score(sample(centers, radius))
        if score > best_score:
            best_score, best = score, (centers, grid_point, radius)
        cycle = cycle[1:] + cycle[:1]

    if best is None or best_score < _MIN_ORIENTATION_SCORE:
        return None
    centers, grid_point, radius = best

    # Neutral 8 (#19) and Neutral 6.5 (#20): mid-grey patches for raw white
    # balance - not the white patch (clips) or black (noisy). Inner ~40% of
    # each, as an axis-aligned box built from the patch's own step vectors so
    # any chart rotation works.
    neutral_boxes_norm: List[Tuple[float, float, float, float]] = []
    for idx in ((rows - 1) * cols + 1, (rows - 1) * cols + 2):
        r, c = divmod(idx, cols)
        u, v = (c + 0.5) / cols, (r + 0.5) / rows
        center = grid_point(u, v)
        half_u = (grid_point(u + 0.5 / cols, v) - grid_point(u - 0.5 / cols, v)) * 0.2
        half_v = (grid_point(u, v + 0.5 / rows) - grid_point(u, v - 0.5 / rows)) * 0.2
        pts = np.array([
            center + half_u + half_v, center + half_u - half_v,
            center - half_u + half_v, center - half_u - half_v,
        ])
        neutral_boxes_norm.append((
            float(pts[:, 0].min()) / w, float(pts[:, 1].min()) / h,
            float(pts[:, 0].max()) / w, float(pts[:, 1].max()) / h,
        ))

    return {
        "centers": centers,
        "neutral_boxes_norm": neutral_boxes_norm,
        "suggested_radius": radius,
        "orientation_score": best_score,
    }


def detect_colorchecker(
    img_rgb: np.ndarray, *, rows: int = 4, cols: int = 6
) -> Optional[Dict[str, Any]]:
    """
    Auto-detect a ColorChecker Classic anywhere in ``img_rgb`` (uint8 RGB).

    Returns ``None`` if no chart is found, else a dict with:
      ``centers``            (24, 2) float patch centres in pixel coords, in
                             REFERENCE_SRGB order (dark skin -> black).
      ``neutral_boxes_norm`` (x0, y0, x1, y1) boxes (fractions of W/H) over the
                             Neutral 8 and Neutral 6.5 patches, for raw white
                             balance sampling.
      ``suggested_radius``   patch-size-relative sampling half-width (px).
      ``orientation_score``  reference-layout correlation of the chosen
                             rotation (max 3.0).

    Patch centres come from bilinearly interpolating the detected chart box
    over a rows x cols grid, after resolving which of the four 90-degree
    rotations the chart sits at (see ``_oriented_chart_grid``) - so a portrait
    shot, whose calibration render is deliberately unrotated, still maps
    correctly. The centres are geometry, valid on any co-registered render
    (the linear CCM render, the raw CFA).
    """
    if cv2 is None:
        raise RuntimeError(
            f"ColorChecker auto-detection needs OpenCV, which isn't available "
            f"({_CV2_IMPORT_ERROR}); install `opencv-contrib-python-headless`"
        )
    if not hasattr(cv2, "mcc"):
        raise RuntimeError(
            "this OpenCV build has no `mcc` module; the ColorChecker detector "
            "ships in opencv-contrib - install `opencv-contrib-python-headless` "
            "(replacing `opencv-python-headless`), or pass explicit `patch_centers`"
        )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    detector = cv2.mcc.CCheckerDetector_create()
    if not detector.process(bgr, cv2.mcc.MCC24):
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None

    box = np.asarray(checkers[0].getBox(), dtype=np.float32).reshape(-1, 2)
    if box.shape[0] != 4:
        return None
    return _oriented_chart_grid(img_rgb, box, rows=rows, cols=cols)


class PatchSampler:
    """Locate and sample the 24 ColorChecker patches from an RGB image."""

    @staticmethod
    def auto_sample(img_rgb: np.ndarray, grid: Tuple[int, int] = (4, 6)) -> np.ndarray:
        """
        Divide the image into a (rows x cols) grid and sample the centre of each
        cell. Works well only when the ColorChecker fills the frame and is
        reasonably upright; for anything else pass explicit ``patch_centers``.

        Returns (N, 3) float32 linear-light RGB.
        """
        rows, cols = grid
        h, w = img_rgb.shape[:2]
        # Shrink sampling box slightly to avoid dark borders
        margin_y = int(h * 0.05)
        margin_x = int(w * 0.05)
        cell_h = (h - 2 * margin_y) // rows
        cell_w = (w - 2 * margin_x) // cols

        samples = []
        for r in range(rows):
            for c in range(cols):
                cy = margin_y + r * cell_h + cell_h // 2
                cx = margin_x + c * cell_w + cell_w // 2
                # Sample a small region and median to reduce noise
                patch = img_rgb[cy - 10:cy + 10, cx - 10:cx + 10].reshape(-1, 3)
                samples.append(np.median(patch, axis=0))

        measured_srgb = np.array(samples, dtype=np.float32) / 255.0
        return _srgb_to_linear(measured_srgb)

    @staticmethod
    def sample_at_centers(
        img_rgb: np.ndarray,
        centers: Sequence[Tuple[int, int]],
        radius: int = 10,
    ) -> np.ndarray:
        """
        Sample patches at explicit pixel (x, y) centres - the reliable path when
        the chart does not fill the frame.

        Parameters
        ----------
        img_rgb : (H, W, 3) uint8
        centers : 24 (x, y) pixel coords, in the same order as REFERENCE_SRGB
        radius  : half-side of the square sampling region

        Returns (24, 3) float32 linear-light RGB.
        """
        samples = []
        for x, y in centers:
            x, y = int(x), int(y)
            patch = img_rgb[y - radius:y + radius, x - radius:x + radius].reshape(-1, 3)
            samples.append(np.median(patch, axis=0))
        measured_srgb = np.array(samples, dtype=np.float32) / 255.0
        return _srgb_to_linear(measured_srgb)

    @staticmethod
    def sample_linear_at_centers(
        img_linear: np.ndarray,
        centers: Sequence[Tuple[float, float]],
        radius: int = 10,
    ) -> np.ndarray:
        """
        Sample patches from an already-**linear-light** float RGB image at the
        given (x, y) centres - the precise path when calibrating from a 16-bit
        linear RAW render (no sRGB round-trip). Returns (N, 3) float32 linear RGB.
        """
        h, w = img_linear.shape[:2]
        samples = []
        for x, y in centers:
            x, y = int(round(float(x))), int(round(float(y)))
            x0, y0 = max(0, x - radius), max(0, y - radius)
            patch = img_linear[y0:y + radius, x0:x + radius].reshape(-1, 3)
            samples.append(np.median(patch, axis=0))
        return np.asarray(samples, dtype=np.float32)


class ColorCorrector:
    """
    Holds a fitted 3x3 Color Correction Matrix and applies it to RGB images.

    All image math lives here, decoupled from Viam, so the same logic can be
    unit-tested or driven from a script. Operates on numpy uint8 RGB arrays;
    callers convert to/from PIL or base64 at the edges.
    """

    def __init__(self, ccm: np.ndarray):
        ccm = np.asarray(ccm, dtype=np.float32)
        if ccm.shape != (3, 3):
            raise ValueError(f"CCM must be a 3x3 matrix, got shape {ccm.shape}")
        self.ccm = ccm

    @classmethod
    def identity(cls) -> "ColorCorrector":
        """A no-op corrector (passes colors through unchanged)."""
        return cls(np.eye(3, dtype=np.float32))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate_from_rgb(
        cls,
        img_rgb: np.ndarray,
        patch_centers: Optional[Sequence[Tuple[int, int]]] = None,
        radius: int = 10,
    ) -> "ColorCorrector":
        """
        Fit a CCM from an RGB image of the ColorChecker Classic.

        If ``patch_centers`` is given (24 (x, y) coords in REFERENCE_SRGB order)
        the patches are sampled there; otherwise auto-sampling assumes the chart
        fills the frame.
        """
        if patch_centers is not None:
            measured_linear = PatchSampler.sample_at_centers(img_rgb, patch_centers, radius)
        else:
            measured_linear = PatchSampler.auto_sample(img_rgb)

        reference_linear = _srgb_to_linear(REFERENCE_SRGB)
        ccm = _fit_ccm(measured_linear, reference_linear)
        return cls(ccm)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.ccm, np.eye(3, dtype=np.float32)))

    def apply_to_linear(self, img_linear: np.ndarray) -> np.ndarray:
        """
        Apply the CCM to an (H, W, 3) **linear-light** float RGB array, returning
        a linear float array. This is the high-precision path: callers working
        from 16-bit RAW stay in linear float end to end and only encode the
        output transfer curve at export. Identity is a no-op passthrough.
        """
        if self.is_identity:
            return img_linear
        h, w = img_linear.shape[:2]
        corrected = (img_linear.reshape(-1, 3) @ self.ccm.T).reshape(h, w, 3)
        return corrected.astype(np.float32)

    def apply_to_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply the CCM to an (H, W, 3) uint8 sRGB array, returning uint8 sRGB.

        Convenience wrapper for the 8-bit streaming path (proxied JPEG/PNG
        frames): sRGB -> linear -> CCM -> sRGB. A no-op (identity) matrix returns
        the input untouched, avoiding gamma round-trip rounding.
        """
        if self.is_identity:
            return img_rgb
        img_linear = _srgb_to_linear(img_rgb.astype(np.float32) / 255.0)
        corrected_srgb = _linear_to_srgb(self.apply_to_linear(img_linear))
        return np.rint(corrected_srgb * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def delta_e_report(
        self,
        img_rgb: np.ndarray,
        patch_centers: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> dict:
        """
        Per-patch color error before vs. after correction, as a quick measure of
        calibration quality. Distances are Euclidean in linear RGB (a rough ΔE
        proxy, not CIE ΔE), scaled by 100.
        """
        if patch_centers is not None:
            measured = PatchSampler.sample_at_centers(img_rgb, patch_centers)
        else:
            measured = PatchSampler.auto_sample(img_rgb)
        ref_linear = _srgb_to_linear(REFERENCE_SRGB)
        corrected_linear = (measured @ self.ccm.T).clip(0, 1)

        def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum((a - b) ** 2, axis=1)) * 100

        before = dist(measured, ref_linear)
        after = dist(corrected_linear, ref_linear)
        return {
            "before": {"mean": float(before.mean()), "max": float(before.max())},
            "after": {"mean": float(after.mean()), "max": float(after.max())},
        }


# ---------------------------------------------------------------------------
# Image <-> base64 / PIL helpers (used on the DoCommand boundary)
# ---------------------------------------------------------------------------

def _base64_to_rgb(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded image (JPEG/PNG) into an (H, W, 3) uint8 RGB array."""
    raw = base64.b64decode(image_base64)
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(pil)


class ColorCorrection(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "image-processing"), "color-correction"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Create a new instance of this Camera component.

        ``EasyResource.new`` only constructs the instance - it does *not* call
        ``reconfigure``, and viam-server only calls ``reconfigure`` on later
        config changes, not on the initial add. So we must configure here, or
        ``self.camera``/``self.corrector`` won't exist when the first request
        arrives.
        """
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate config and declare the source camera as a required dependency."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        if not camera:
            raise ValueError("Missing required attribute `camera` in config")

        ccm = attrs.get("ccm")
        if ccm is not None and np.array(ccm, dtype=np.float32).shape != (3, 3):
            raise ValueError("`ccm` must be a 3x3 matrix")

        formats = attrs.get("output_formats")
        if formats is not None:
            unknown = [f for f in formats if f not in EXPORT_FORMATS]
            if unknown:
                raise ValueError(
                    f"unknown `output_formats` {unknown}; valid: "
                    f"{sorted(EXPORT_FORMATS)}"
                )

        output_dir = attrs.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError("`output_dir` must be a string path")

        return [str(camera)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Wire up the source camera and load the CCM from the `ccm` attribute."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        source = dependencies.get(Camera.get_resource_name(str(camera)))
        if source is None:
            raise ValueError(f"Could not resolve source camera dependency `{camera}`")
        self.camera: Camera = source

        # The CCM lives inline in the `ccm` config attribute. Run the
        # `calibrate_color` DoCommand to compute one, then copy the returned
        # matrix into this attribute to make the correction persist.
        ccm = attrs.get("ccm")
        if ccm is not None:
            self.corrector = ColorCorrector(np.array(ccm, dtype=np.float32))
        else:
            self.corrector = ColorCorrector.identity()
            self.logger.info("No `ccm` configured; passing images through uncorrected")

        # Studio export settings (used by the `capture` DoCommand RAW pipeline).
        # output_dir defaults to wherever the source file was downloaded.
        self._output_dir: Optional[str] = attrs.get("output_dir") or None
        self._output_formats: List[str] = list(
            attrs.get("output_formats") or DEFAULT_OUTPUT_FORMATS
        )
        self._jpeg_quality: int = int(attrs.get("jpeg_quality", 95))
        self._white_balance = attrs.get("white_balance", "camera")
        self._write_sidecar: bool = bool(attrs.get("write_sidecar", True))

        # The `upload` DoCommand authenticates to the cloud with the API key
        # Viam injects into every module process (VIAM_API_KEY / VIAM_API_KEY_ID),
        # so no credentials are configured here. part_id falls back to the
        # machine's env var. The data client is created lazily and reused.
        self._part_id: Optional[str] = (
            attrs.get("part_id") or os.environ.get("VIAM_MACHINE_PART_ID") or None
        )
        self._data_client: Optional[DataClient] = None

        # In-flight deferred captures (`capture` with `defer: true`), keyed by
        # the capture_id handed back to the caller. Preserved across
        # reconfigure so a mid-sequence config change doesn't orphan results.
        self._pending_captures: Dict[str, "asyncio.Task"] = getattr(
            self, "_pending_captures", {}
        )
        self._capture_seq: int = getattr(self, "_capture_seq", 0)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _correct_viam_image(self, image: ViamImage) -> ViamImage:
        """Apply the CCM to a single ViamImage, preserving its mime type."""
        pil_image = viam_to_pil_image(image).convert("RGB")
        corrected = self.corrector.apply_to_rgb(np.array(pil_image))
        mime = image.mime_type if image.mime_type == CameraMimeType.PNG else CameraMimeType.JPEG
        return pil_to_viam_image(Image.fromarray(corrected), mime)

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        images, metadata = await self.camera.get_images(
            filter_source_names=filter_source_names,
            extra=extra,
            timeout=timeout,
            **kwargs,
        )

        corrected: List[NamedImage] = []
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                viam_img = self._correct_viam_image(image)
                corrected.append(NamedImage(image.name, viam_img.data, viam_img.mime_type))
            else:
                # Pass non-image payloads (e.g. depth) through untouched.
                self.logger.debug(
                    f"Passing through image '{image.name}' with uncorrectable "
                    f"mime type {image.mime_type}"
                )
                corrected.append(image)

        return corrected, metadata

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        # Color correction doesn't apply to point clouds; proxy the source.
        return await self.camera.get_point_cloud(extra=extra, timeout=timeout, **kwargs)

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        # Report the source camera's properties; we don't change resolution,
        # mime types, or intrinsics.
        return await self.camera.get_properties(timeout=timeout, **kwargs)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        resp: Dict[str, ValueTypes] = {}

        if "calibrate_color" in command:
            resp["calibrate_color"] = await self._calibrate_color(
                command.get("calibrate_color") or {}, timeout
            )

        if "capture" in command:
            resp["capture"] = await self._capture_corrected(
                command.get("capture") or {}, timeout
            )

        if "capture_result" in command:
            resp["capture_result"] = await self._capture_result(
                command.get("capture_result") or {}
            )

        if "develop" in command:
            resp["develop"] = await self._develop(command.get("develop") or {})

        if "upload" in command:
            resp["upload"] = await self._upload(command.get("upload") or {})

        if not resp:
            raise ValueError(
                "no recognized command; supported: calibrate_color, capture, "
                "capture_result, develop, upload"
            )
        return resp

    # ------------------------------------------------------------------
    # DoCommand handlers
    # ------------------------------------------------------------------

    def _linear_from_capture_response(
        self,
        capture: Any,
        white_balance: Any,
        exposure_stops: float,
        half_size: bool = False,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Turn a source camera's ``capture`` DoCommand response into a
        **linear-light** float RGB array (sRGB primaries) plus the source path.

        Two shapes are supported, in priority order:

        * ``image_base64`` - an inline JPEG/PNG, small enough to ship over gRPC
          (the Canon CCAPI flow). Decoded as 8-bit sRGB and linearized; ``None``
          path since there's no file on disk.
        * ``saved_to`` (or ``path``) - a file the source wrote to disk. This is
          the PTP model's handoff for full-resolution stills, including RAW
          (CR3/NEF/ARW/...). It's demosaiced to 16-bit linear by
          ``image_io.load_linear_rgb`` (applying white balance / exposure at the
          raw stage), with no precision lost before color correction.
        """
        if not isinstance(capture, Mapping):
            raise ValueError("source camera `capture` returned an unexpected response")

        image_b64 = capture.get("image_base64")
        if image_b64:
            rgb8 = _base64_to_rgb(image_b64).astype(np.float32) / 255.0
            return srgb_to_linear(rgb8).astype(np.float32), None

        path = capture.get("saved_to") or capture.get("path")
        if path:
            linear = load_linear_rgb(
                str(path), white_balance=white_balance,
                exposure_stops=exposure_stops, half_size=half_size,
            )
            return linear, str(path)

        raise ValueError(
            "source camera `capture` returned neither an `image_base64` field "
            "nor a `saved_to` path; if the source is the PTP camera, configure "
            "its `download_dir` so captures are written to disk"
        )

    async def _acquire_calibration_source(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Get the source to calibrate from, as ``(raw_path, rgb8)``.

        ``raw_path`` is set when a RAW file is on disk - the path that unlocks
        raw-CFA white balance. Otherwise ``rgb8`` is an 8-bit sRGB frame for
        CCM-only calibration. Exactly one is non-None.

        Resolution order: explicit ``path`` -> ``use_capture`` (trigger a still
        on the source, prefer its ``saved_to`` RAW) -> the streaming frame.
        """
        def _file_to_rgb8(p: str) -> np.ndarray:
            linear = load_linear_rgb(str(p), white_balance="camera")
            return (linear_to_srgb(linear) * 255.0).clip(0, 255).astype(np.uint8)

        path = opts.get("path")
        if path:
            if is_raw(str(path)):
                return str(path), None
            return None, _file_to_rgb8(str(path))

        if opts.get("use_capture"):
            capture_opts = opts.get("capture_options", {"af": True})
            source_resp = await self.camera.do_command(
                {"capture": capture_opts}, timeout=timeout
            )
            capture = source_resp.get("capture", source_resp)
            if isinstance(capture, Mapping):
                p = capture.get("saved_to") or capture.get("path")
                if p and is_raw(str(p)):
                    return str(p), None
                if p:
                    return None, _file_to_rgb8(str(p))
                b64 = capture.get("image_base64")
                if b64:
                    return None, _base64_to_rgb(b64)
            raise ValueError(
                "source `capture` returned nothing usable for calibration "
                "(no RAW `saved_to`, file `path`, or inline `image_base64`)"
            )

        images, _ = await self.camera.get_images(timeout=timeout)
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                return None, np.array(viam_to_pil_image(image).convert("RGB"))
        raise ValueError("source camera returned no JPEG/PNG image to use")

    async def _calibrate_color(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Auto-calibrate from a ColorChecker frame: detect the chart (cv2.mcc),
        measure white balance from the raw CFA, and fit a CCM under that same
        white balance. Both are applied to this component immediately and
        returned so they can be copied into the ``ccm`` / ``white_balance``
        config attributes to persist across restarts.

        Options (all optional):
          ``use_capture``    trigger a full-res still on the source (needed for
                             white balance - the RAW must be on disk).
          ``path``           calibrate from a RAW/image file already on disk.
          ``capture_options``forwarded to the source ``capture`` (e.g. {"af": true}).
          ``compute_wb``     derive white balance from the chart (default true).
          ``patch_centers``  24 [x, y] coords to override auto-detection.
          ``radius``         patch sampling half-width in px (default: ~15% of
                             the detected patch size, or 10 with manual centers).

        Returns ``ccm`` (pure-colour, ~unity-gain), ``white_balance``
        ([r,g,b,g2] or null), ``exposure_stops`` (the brightness offset the chart
        implied vs. the reference - pass it back as ``exposure_stops`` on
        capture/develop to render at the ColorChecker's nominal brightness), and
        a ``delta_e`` report whose ``after`` figure is exposure-normalised colour
        accuracy.
        """
        raw_path, rgb8 = await self._acquire_calibration_source(opts, timeout)
        compute_wb = bool(opts.get("compute_wb", True))
        radius = int(opts["radius"]) if "radius" in opts else None
        manual_centers = opts.get("patch_centers")

        # Image used to *locate* the patches. For a RAW we render it unrotated
        # (user_flip=0) so the centres map straight onto the CFA and the linear
        # CCM render below.
        detect_img = render_raw_for_detection(raw_path) if raw_path else rgb8

        if manual_centers is not None:
            centers = np.array(
                [(int(x), int(y)) for x, y in manual_centers], dtype=np.float32
            )
            neutral_boxes = None
            radius = radius if radius is not None else 10
        else:
            detection = detect_colorchecker(detect_img)
            if detection is None:
                raise ValueError(
                    "could not auto-detect the ColorChecker; ensure the whole "
                    "chart is visible and unobstructed, or pass `patch_centers`"
                )
            if radius is None:
                radius = int(detection["suggested_radius"])
            centers = detection["centers"]
            neutral_boxes = detection["neutral_boxes_norm"]

        # White balance from the raw Bayer/CFA under the neutral patches.
        wb: Optional[List[float]] = None
        wb_note: Optional[str] = None
        if compute_wb:
            if raw_path and neutral_boxes:
                wb = compute_raw_wb_multipliers(raw_path, neutral_boxes)
            elif raw_path:
                wb_note = (
                    "skipped: white balance needs auto-detected neutral patches; "
                    "omit `patch_centers` to enable it"
                )
            else:
                wb_note = (
                    "skipped: raw-CFA white balance needs a RAW capture - set "
                    "`use_capture: true` with a RAW source, or pass a RAW `path`"
                )

        # Fit the CCM on patches developed with the SAME white balance the
        # captures will use, so the matrix and the WB stay consistent.
        reference_linear = _srgb_to_linear(REFERENCE_SRGB)
        if raw_path:
            linear = load_linear_rgb(
                raw_path,
                white_balance=(wb if wb is not None else "camera"),
                user_flip=0,  # match detect_img so `centers` line up
            )
            measured_linear = PatchSampler.sample_linear_at_centers(linear, centers, radius)
        else:
            measured_linear = PatchSampler.sample_at_centers(
                detect_img, [(int(x), int(y)) for x, y in centers], radius
            )

        # Decouple exposure (a single scalar) from colour (the matrix). The chart
        # is typically exposed below the reference's nominal brightness; fitting
        # the CCM directly makes it absorb that gain (diagonal >> 1), which then
        # brightens *every* developed frame and clips highlights early. Instead we
        # scale the measured patches so the neutral ramp matches the reference
        # luminance, fit the CCM on that (keeping it ~unity-gain, pure colour),
        # and report the implied exposure offset for the caller to dial in via
        # `exposure_stops`. Neutral 8 / 6.5 / 5 / 3.5 - skip the clip-prone white
        # and noisy black ends of the ramp.
        neutral_fit = [19, 20, 21, 22]
        meas_neutral = measured_linear[neutral_fit].reshape(-1)
        ref_neutral = reference_linear[neutral_fit].reshape(-1)
        energy = float(np.dot(meas_neutral, meas_neutral))
        exposure_scale = float(np.dot(meas_neutral, ref_neutral) / energy) if energy > 0 else 1.0
        measured_fit = measured_linear * exposure_scale

        ccm = _fit_ccm(measured_fit, reference_linear)
        corrector = ColorCorrector(ccm)

        def _delta_e_stats(values: np.ndarray) -> Dict[str, float]:
            d = np.sqrt(np.sum((np.clip(values, 0, 1) - reference_linear) ** 2, axis=1)) * 100
            return {"mean": float(d.mean()), "max": float(d.max())}

        # "after" is exposure-normalised, so it reflects pure colour accuracy
        # independent of how bright you choose to render (via exposure_stops).
        report = {
            "before": _delta_e_stats(measured_linear),
            "after": _delta_e_stats(measured_fit @ ccm.T),
        }
        exposure_stops = float(np.log2(exposure_scale)) if exposure_scale > 0 else 0.0

        self.corrector = corrector
        if wb is not None:
            # Subsequent capture/develop default to this WB unless overridden.
            self._white_balance = wb

        self.logger.info(
            f"Calibrated CCM (delta-E mean {report['before']['mean']:.1f} -> "
            f"{report['after']['mean']:.1f}, exposure {exposure_stops:+.2f} stops)"
            + (f"; white balance [{', '.join(f'{v:.3f}' for v in wb)}]" if wb else "")
            + "; copy `ccm`"
            + (" and `white_balance`" if wb else "")
            + " into the component config to persist"
        )
        result: Dict[str, ValueTypes] = {
            "ccm": corrector.ccm.tolist(),
            "white_balance": wb,
            "exposure_stops": exposure_stops,
            "delta_e": report,
        }
        if wb_note:
            result["white_balance_note"] = wb_note
        return result

    async def _capture_corrected(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Studio capture: trigger a full-resolution still on the source camera,
        develop it through the 16-bit linear pipeline (white balance + CCM),
        and write rendered exports - leaving any RAW original untouched.

        ``opts`` (all optional):
          ``capture_options``  forwarded to the source's ``capture`` (e.g. {"af": true})
          ``white_balance``    "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops``   exposure compensation applied at the raw stage
          ``output_formats``   subset of tiff16/tiff8/jpeg/png16/png8; pass []
                               to skip exports (preview-only capture - develop
                               the RAW later with the ``develop`` command)
          ``output_dir``       where to write exports (default: next to the source file)
          ``defer``            true -> return as soon as the shutter has fired
                               (the rig is free to move); download/decode/preview
                               continue in the background and are fetched with
                               ``capture_result``. Requires a source camera with
                               a ``trigger`` DoCommand (the ptp model). Deferred
                               captures never write exports or a sidecar - run
                               ``develop`` on the RAW for those.

        Returns the written export paths, the sidecar path, and a small base64
        JPEG preview (not the full-res image - that stays on disk). With
        ``defer`` it instead returns {"capture_id", "status": "pending",
        "camera_path"} immediately after the shutter fires.
        """
        if opts.get("defer"):
            return await self._capture_deferred(opts, timeout)

        capture_opts = opts.get("capture_options", {"af": True})
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", 0.0))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir
        # When nothing is being exported, the decode only feeds the preview -
        # a half-resolution demosaic is ~4x faster and indistinguishable there.
        preview_only = not formats

        start = time.perf_counter()
        source_resp = await self.camera.do_command({"capture": capture_opts}, timeout=timeout)
        capture = source_resp.get("capture", source_resp)
        self.logger.debug(
            f"[timing] source camera capture (incl. download): "
            f"{time.perf_counter() - start:.2f}s"
        )

        t_decode = time.perf_counter()
        # The decode and develop/export steps are seconds of pure CPU; run them
        # in a worker thread so the event loop keeps serving other requests.
        linear, source_path = await asyncio.to_thread(
            self._linear_from_capture_response,
            capture, white_balance, exposure_stops, preview_only,
        )
        self.logger.debug(
            f"[timing] decode to linear RGB (incl. white balance): "
            f"{time.perf_counter() - t_decode:.2f}s"
        )
        result = await asyncio.to_thread(
            self._develop_one,
            linear, source_path, white_balance, exposure_stops, formats,
            out_dir_override,
        )
        self.logger.debug(
            f"[timing] capture pipeline total: {time.perf_counter() - start:.2f}s"
        )
        return result

    async def _capture_deferred(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Fire the shutter and return as soon as the exposure is done, so the
        caller can move the rig while the slow parts (USB download, demosaic,
        preview encode) run in a background task. The source's lock serializes
        camera access, so a background download naturally queues ahead of the
        next pose's trigger.
        """
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", 0.0))

        start = time.perf_counter()
        try:
            resp = await self.camera.do_command(
                {"trigger": opts.get("capture_options", {})}, timeout=timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"deferred capture needs a source camera with a `trigger` "
                f"DoCommand (the ptp model); triggering failed: {exc}"
            ) from exc
        trig = resp.get("trigger") or {}
        camera_path = trig.get("path")
        if not camera_path:
            raise ValueError("source camera `trigger` returned no `path`")
        self.logger.debug(
            f"[timing] deferred capture trigger (shutter + settle): "
            f"{time.perf_counter() - start:.2f}s"
        )

        self._capture_seq += 1
        stem = os.path.splitext(os.path.basename(str(camera_path)))[0]
        capture_id = f"{self._capture_seq}-{stem}"
        self._pending_captures[capture_id] = asyncio.create_task(
            self._finish_deferred_capture(
                capture_id, str(camera_path), white_balance, exposure_stops
            )
        )
        # Drop completed-and-collected stragglers if a caller never fetched
        # them, so an unattended sequence can't grow the table without bound.
        if len(self._pending_captures) > 64:
            for key in [
                k for k, t in self._pending_captures.items() if t.done()
            ][:-64]:
                self._pending_captures.pop(key, None)

        return {
            "capture_id": capture_id,
            "status": "pending",
            "camera_path": str(camera_path),
        }

    async def _finish_deferred_capture(
        self,
        capture_id: str,
        camera_path: str,
        white_balance: Any,
        exposure_stops: float,
    ) -> Dict[str, ValueTypes]:
        """Background half of a deferred capture: download the still from the
        camera, decode at half size, apply the CCM, and build the preview. No
        exports or sidecar - the RAW on disk is the handoff to ``develop``."""
        start = time.perf_counter()
        resp = await self.camera.do_command({"download": {"path": camera_path}})
        meta = resp.get("download") or {}
        saved = meta.get("saved_to")
        if not saved:
            raise ValueError(
                f"source camera did not save {camera_path!r} to disk; configure "
                f"its `download_dir` so deferred captures can be developed later"
            )
        linear = await asyncio.to_thread(
            load_linear_rgb, str(saved),
            white_balance=white_balance, exposure_stops=exposure_stops,
            half_size=True,
        )
        corrected = await asyncio.to_thread(self.corrector.apply_to_linear, linear)
        preview = await asyncio.to_thread(linear_to_jpeg_base64, corrected)
        self.logger.debug(
            f"[timing] deferred capture {capture_id} background "
            f"(download + decode + preview): {time.perf_counter() - start:.2f}s"
        )
        return {
            "capture_id": capture_id,
            "status": "done",
            "source_path": str(saved),
            "image_base64": preview,
            "mime_type": CameraMimeType.JPEG.value,
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
        }

    async def _capture_result(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Fetch the result of a deferred capture.

        ``opts``:
          ``id``        the capture_id returned by ``capture`` with ``defer`` (required)
          ``wait_sec``  how long to wait for the background work (default 60;
                        0 polls). Returns {"status": "pending"} on timeout -
                        call again to keep waiting.
        """
        capture_id = opts.get("id") or opts.get("capture_id")
        if not capture_id:
            raise ValueError(
                "`capture_result` needs the `id` returned by `capture` with `defer`"
            )
        capture_id = str(capture_id)
        task = self._pending_captures.get(capture_id)
        if task is None:
            raise ValueError(
                f"unknown capture id {capture_id!r}; it may have already been "
                f"collected, or the module restarted since the capture"
            )
        wait_sec = float(opts.get("wait_sec", 60.0))
        try:
            # shield() so a timeout here doesn't cancel the background work.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_sec)
        except asyncio.TimeoutError:
            return {"capture_id": capture_id, "status": "pending"}
        except Exception as exc:
            self._pending_captures.pop(capture_id, None)
            raise RuntimeError(f"deferred capture {capture_id} failed: {exc}") from exc
        self._pending_captures.pop(capture_id, None)
        return result

    async def _develop(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Develop existing image file(s) already on disk - no camera trigger.
        Point this at a RAW (CR3/NEF/ARW/...) or any JPEG/PNG/TIFF and it runs
        the same 16-bit linear pipeline (white balance + CCM) and writes the
        rendered exports + sidecar, leaving the original untouched.

        ``opts``:
          ``path``           a single file path (returns that file's result), OR
          ``paths``          a list of file paths (returns {"developed": [...]})
          ``white_balance``  "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops`` exposure compensation applied at the raw stage
          ``output_formats`` subset of tiff16/tiff8/jpeg/png16/png8
          ``output_dir``     where to write exports (default: next to each file)
        """
        raw_paths = opts.get("paths")
        single = raw_paths is None
        if single:
            one = opts.get("path")
            if not one:
                raise ValueError(
                    "`develop` needs a `path` (string) or `paths` (list of strings)"
                )
            raw_paths = [one]
        paths = [str(p) for p in raw_paths]

        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", 0.0))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir

        results: List[Mapping[str, ValueTypes]] = []
        for path in paths:
            t_file = time.perf_counter()
            # Decode + export are seconds of pure CPU per file; keep them off
            # the event loop so other requests stay responsive mid-batch.
            linear = await asyncio.to_thread(
                load_linear_rgb,
                path, white_balance=white_balance, exposure_stops=exposure_stops,
            )
            results.append(
                await asyncio.to_thread(
                    self._develop_one,
                    linear, path, white_balance, exposure_stops, formats,
                    out_dir_override,
                    # Skip the per-file base64 preview in batch mode to keep the
                    # response small; a single develop still returns its preview.
                    include_preview=single,
                )
            )
            self.logger.debug(
                f"[timing] develop {os.path.basename(path)} total: "
                f"{time.perf_counter() - t_file:.2f}s"
            )

        if single:
            return results[0]
        return {"developed": results, "count": len(results)}

    def _develop_one(
        self,
        linear: np.ndarray,
        source_path: Optional[str],
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        out_dir_override: Optional[str],
        include_preview: bool = True,
    ) -> Dict[str, ValueTypes]:
        """
        Shared core for ``capture`` and ``develop``: apply the CCM in linear
        light, write the rendered exports (non-destructively) and a sidecar, and
        return the result. ``linear`` is linear-light float RGB; ``source_path``
        is the originating file (or None for an inline base64 capture).
        """
        t_ccm = time.perf_counter()
        corrected = self.corrector.apply_to_linear(linear)
        self.logger.debug(
            f"[timing] apply color correction (CCM): {time.perf_counter() - t_ccm:.2f}s"
        )

        # Exports land alongside the source file unless an output_dir is set.
        out_dir = out_dir_override or (
            os.path.dirname(source_path) if source_path else None
        )
        stem = (
            os.path.splitext(os.path.basename(source_path))[0]
            if source_path else "capture"
        )
        # A RAW source (.cr3/.nef/...) never collides with our .tif/.jpg/.png
        # exports, so its name is preserved. But if the source is itself a
        # JPEG/PNG/TIFF, a same-name export would overwrite the original - so
        # suffix the exports to keep the pipeline non-destructive.
        if source_path and not is_raw(source_path):
            stem = stem + "_corrected"
        exports: Dict[str, str] = {}
        if out_dir:
            t_export = time.perf_counter()
            exports = export_renditions(
                corrected, out_dir, stem, formats, quality=self._jpeg_quality
            )
            self.logger.debug(
                f"[timing] export {len(exports)} format(s): "
                f"{time.perf_counter() - t_export:.2f}s"
            )
            self.logger.info(f"exported {list(exports)} for {stem} to {out_dir}")
        else:
            self.logger.warning(
                "no `output_dir` configured and source has no path; "
                "returning a preview only (nothing written to disk)"
            )

        sidecar = None
        if self._write_sidecar and source_path:
            sidecar = self._write_sidecar_file(
                source_path, white_balance, exposure_stops, formats, exports
            )

        result: Dict[str, ValueTypes] = {
            "source_path": source_path,
            "exports": exports,
            "sidecar": sidecar,
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
        }
        if include_preview:
            result["image_base64"] = linear_to_jpeg_base64(corrected)
            result["mime_type"] = CameraMimeType.JPEG.value
        return result

    def _write_sidecar_file(
        self,
        source_path: str,
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        exports: Mapping[str, str],
    ) -> str:
        """
        Write a ``<stem>.json`` sidecar next to the (untouched) source file
        recording exactly how it was developed - the non-destructive record that
        lets a capture be reproduced or re-exported later.
        """
        sidecar_path = os.path.splitext(source_path)[0] + ".json"
        record = {
            "source": os.path.basename(source_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "white_balance": white_balance,
            "exposure_stops": exposure_stops,
            "ccm": self.corrector.ccm.tolist(),
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
            "output_formats": list(formats),
            "exports": {k: os.path.basename(v) for k, v in exports.items()},
        }
        with open(sidecar_path, "w") as f:
            json.dump(record, f, indent=2)
        return sidecar_path

    async def _get_data_client(self) -> DataClient:
        """
        Lazily build (and cache) a cloud ``DataClient`` from the API key Viam
        injects into the module process (``VIAM_API_KEY`` / ``VIAM_API_KEY_ID``).

        We dial the app channel directly rather than via
        ``ViamClient.create_from_env_vars``: in viam-sdk 0.77.0 that path
        authenticates the channel inside ``_dial_app`` and then authenticates a
        *second* time, which the server rejects with "already authenticated;
        cannot re-authenticate". ``_dial_app`` alone performs the single, correct
        auth, and the resulting channel carries the bearer token we hand to the
        ``DataClient``.
        """
        if self._data_client is not None:
            return self._data_client
        api_key = os.environ.get("VIAM_API_KEY")
        api_key_id = os.environ.get("VIAM_API_KEY_ID")
        if not api_key or not api_key_id:
            raise ValueError(
                "`upload` could not authenticate: VIAM_API_KEY / VIAM_API_KEY_ID "
                "were not present in the module environment. This requires "
                "running on a cloud-connected machine."
            )
        dial_options = DialOptions(
            credentials=Credentials(type="api-key", payload=api_key),
            auth_entity=api_key_id,
        )
        channel = await _dial_app("app.viam.com", dial_options)
        metadata = getattr(channel, "_metadata", {})
        self._data_client = DataClient(channel, metadata)
        return self._data_client

    async def _upload(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Upload files already on disk to Viam, tagged for later retrieval.

        The full-resolution captures (CR3 + the rendered TIFF/JPEG exports + the
        JSON sidecar) live on this machine's filesystem; this ships them straight
        to the cloud so they never have to travel back through the browser. The
        webapp passes every path that shares a capture's filename stem, so a
        single selected shot uploads as a complete set under one SKU tag.

        ``opts``:
          ``paths``          list of file paths on disk to upload (required)
          ``tags``           tags to attach to every uploaded file (e.g. SKU)
          ``part_id``        override the configured / env machine part id
          ``component_name`` camera name to associate the data with (optional)
        """
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`upload` needs a non-empty `paths` list")
        paths = [str(p) for p in raw_paths]
        tags = [str(t) for t in (opts.get("tags") or [])]

        part_id = opts.get("part_id") or self._part_id
        if not part_id:
            raise ValueError(
                "no part id available for upload; set `part_id` in config or pass "
                "it in the command (VIAM_MACHINE_PART_ID was not set)"
            )
        component_name = opts.get("component_name") or self.name

        data_client = await self._get_data_client()

        uploaded: List[str] = []
        failed: List[Dict[str, str]] = []
        for path in paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                ext = os.path.splitext(path)[1]  # e.g. ".cr3", ".tif", ".jpg"
                t_upload = time.perf_counter()
                await data_client.file_upload(
                    part_id=str(part_id),
                    data=data,
                    file_name=os.path.basename(path),
                    file_extension=ext,
                    tags=tags or None,
                    component_type="rdk:component:camera",
                    component_name=str(component_name),
                )
                self.logger.debug(
                    f"[timing] upload {os.path.basename(path)} "
                    f"({len(data) / 1e6:.1f} MB): {time.perf_counter() - t_upload:.2f}s"
                )
                uploaded.append(path)
            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                self.logger.error(f"failed to upload {path}: {exc}")
                failed.append({"path": path, "error": str(exc)})

        self.logger.info(
            f"uploaded {len(uploaded)}/{len(paths)} file(s)"
            + (f" with tags {tags}" if tags else "")
        )
        return {"uploaded": uploaded, "count": len(uploaded), "failed": failed}

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return await self.camera.get_geometries(extra=extra, timeout=timeout)
