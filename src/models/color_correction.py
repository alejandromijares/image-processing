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

import base64
import json
import os
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

from viam.components.camera import Camera
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image
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
    Order 4 chart corners as [top-left, top-right, bottom-right, bottom-left].

    Uses the x+y / x-y heuristic, which is robust for a chart that's roughly
    upright (rotation < ~45deg) - the studio case where you drop the board in
    frame. cv2.mcc already corrects perspective; this just fixes the winding so
    the bilinear grid below lands dark-skin patch first (REFERENCE_SRGB order).
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

    Patch centres come from bilinearly interpolating the detected chart box over
    a rows x cols grid - geometry only, so the same centres are valid on any
    co-registered render (the linear CCM render, the raw CFA).
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
    tl, tr, br, bl = _order_corners(box)

    def grid_point(u: float, v: float) -> np.ndarray:
        top = tl + (tr - tl) * u
        bot = bl + (br - bl) * u
        return top + (bot - top) * v

    centers = np.zeros((rows * cols, 2), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            centers[r * cols + c] = grid_point((c + 0.5) / cols, (r + 0.5) / rows)

    # Neutral 8 (#20) and Neutral 6.5 (#21): bottom row, 2nd and 3rd cells. Avoid
    # the white patch (clips) and black (noisy). Sample the inner ~40% of each.
    h, w = img_rgb.shape[:2]
    half_w = 0.2 * float(np.linalg.norm(tr - tl)) / cols
    half_h = 0.2 * float(np.linalg.norm(bl - tl)) / rows
    neutral_boxes_norm: List[Tuple[float, float, float, float]] = []
    for idx in ((rows - 1) * cols + 1, (rows - 1) * cols + 2):
        cx, cy = centers[idx]
        neutral_boxes_norm.append((
            (cx - half_w) / w, (cy - half_h) / h,
            (cx + half_w) / w, (cy + half_h) / h,
        ))

    return {"centers": centers, "neutral_boxes_norm": neutral_boxes_norm}


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
        return (corrected_srgb * 255.0).clip(0, 255).astype(np.uint8)

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

        if "develop" in command:
            resp["develop"] = await self._develop(command.get("develop") or {})

        if not resp:
            raise ValueError(
                "no recognized command; supported: calibrate_color, capture, develop"
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
                str(path), white_balance=white_balance, exposure_stops=exposure_stops
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
          ``radius``         patch sampling half-width (default 10).

        Returns ``ccm`` (pure-colour, ~unity-gain), ``white_balance``
        ([r,g,b,g2] or null), ``exposure_stops`` (the brightness offset the chart
        implied vs. the reference - pass it back as ``exposure_stops`` on
        capture/develop to render at the ColorChecker's nominal brightness), and
        a ``delta_e`` report whose ``after`` figure is exposure-normalised colour
        accuracy.
        """
        raw_path, rgb8 = await self._acquire_calibration_source(opts, timeout)
        compute_wb = bool(opts.get("compute_wb", True))
        radius = int(opts.get("radius", 10))
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
        else:
            detection = detect_colorchecker(detect_img)
            if detection is None:
                raise ValueError(
                    "could not auto-detect the ColorChecker; ensure the whole "
                    "chart is visible and unobstructed, or pass `patch_centers`"
                )
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
          ``output_formats``   subset of tiff16/tiff8/jpeg/png16/png8
          ``output_dir``       where to write exports (default: next to the source file)

        Returns the written export paths, the sidecar path, and a small base64
        JPEG preview (not the full-res image - that stays on disk).
        """
        capture_opts = opts.get("capture_options", {"af": True})
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", 0.0))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir

        source_resp = await self.camera.do_command({"capture": capture_opts}, timeout=timeout)
        capture = source_resp.get("capture", source_resp)

        linear, source_path = self._linear_from_capture_response(
            capture, white_balance, exposure_stops
        )
        return self._develop_one(
            linear, source_path, white_balance, exposure_stops, formats, out_dir_override
        )

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
            linear = load_linear_rgb(
                path, white_balance=white_balance, exposure_stops=exposure_stops
            )
            results.append(
                self._develop_one(
                    linear, path, white_balance, exposure_stops, formats,
                    out_dir_override,
                    # Skip the per-file base64 preview in batch mode to keep the
                    # response small; a single develop still returns its preview.
                    include_preview=single,
                )
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
        corrected = self.corrector.apply_to_linear(linear)

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
            exports = export_renditions(
                corrected, out_dir, stem, formats, quality=self._jpeg_quality
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

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return await self.camera.get_geometries(extra=extra, timeout=timeout)
