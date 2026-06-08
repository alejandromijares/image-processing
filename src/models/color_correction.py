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

2. DoCommand path - for cameras (like the Canon CCAPI module) whose
   full-resolution stills are exposed through DoCommand rather than the
   streaming ``Images`` method:

       {"capture": {"af": true}}
           -> capture a still from the source camera, color-correct it, and
              return {"image_base64", "mime_type", "path"}.

Calibration:

       {"calibrate_color": {}}
           -> grab a frame of the ColorChecker from the source camera's live
              view, fit a CCM, and return it.
       {"calibrate_color": {"use_capture": true}}
           -> calibrate from a full-resolution still (via the source's capture
              DoCommand) and return the fitted CCM.

`calibrate_color` returns the fitted 3x3 matrix; copy it into the component's
``ccm`` config attribute to make the correction persist across restarts.
"""

import base64
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


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Apply sRGB inverse gamma (gamma-encoded -> linear light)."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma (linear light -> gamma-encoded)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


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

    def apply_to_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply the CCM to an (H, W, 3) uint8 RGB array, returning uint8 RGB.

        Correction is performed in linear light: sRGB -> linear -> CCM -> sRGB.
        A no-op (identity) matrix returns the input untouched, avoiding gamma
        round-trip rounding when no correction is configured.
        """
        if self.is_identity:
            return img_rgb
        img_f = img_rgb.astype(np.float32) / 255.0
        img_linear = _srgb_to_linear(img_f)
        h, w = img_linear.shape[:2]
        corrected = (img_linear.reshape(-1, 3) @ self.ccm.T).reshape(h, w, 3)
        corrected_srgb = _linear_to_srgb(corrected)
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


def _rgb_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 95) -> str:
    """Encode an (H, W, 3) uint8 RGB array to a base64 JPEG string."""
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


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

        if not resp:
            raise ValueError(
                "no recognized command; supported: calibrate_color, capture"
            )
        return resp

    # ------------------------------------------------------------------
    # DoCommand handlers
    # ------------------------------------------------------------------

    async def _acquire_rgb(self, opts: Mapping[str, Any], timeout: Optional[float]) -> np.ndarray:
        """
        Grab a single RGB frame from the source camera for calibration/capture.

        With ``use_capture: true`` it triggers the source's full-resolution still
        via its ``capture`` DoCommand (the Canon CCAPI flow); otherwise it pulls
        the first JPEG/PNG frame from the streaming ``get_images`` path.
        """
        if opts.get("use_capture"):
            capture_opts = opts.get("capture_options", {"af": True})
            source_resp = await self.camera.do_command({"capture": capture_opts}, timeout=timeout)
            capture = source_resp.get("capture", source_resp)
            image_b64 = capture.get("image_base64") if isinstance(capture, Mapping) else None
            if not image_b64:
                raise ValueError(
                    "source camera `capture` did not return an `image_base64` field"
                )
            return _base64_to_rgb(image_b64)

        images, _ = await self.camera.get_images(timeout=timeout)
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                return np.array(viam_to_pil_image(image).convert("RGB"))
        raise ValueError("source camera returned no JPEG/PNG image to use")

    async def _calibrate_color(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Fit a CCM from a ColorChecker frame and apply it to this component
        immediately. The fitted matrix is returned so it can be copied into the
        ``ccm`` config attribute to persist across restarts.

        Options: ``use_capture`` (bool), ``capture_options`` (passed to the
        source capture), ``patch_centers`` (24 [x, y] coords), ``radius`` (int).
        """
        img_rgb = await self._acquire_rgb(opts, timeout)

        patch_centers = opts.get("patch_centers")
        if patch_centers is not None:
            patch_centers = [(int(x), int(y)) for x, y in patch_centers]
        radius = int(opts.get("radius", 10))

        corrector = ColorCorrector.calibrate_from_rgb(img_rgb, patch_centers, radius)
        report = corrector.delta_e_report(img_rgb, patch_centers)
        self.corrector = corrector

        self.logger.info(
            f"Calibrated CCM (delta-E mean {report['before']['mean']:.1f} -> "
            f"{report['after']['mean']:.1f}); copy `ccm` into the component "
            f"config to persist"
        )
        return {
            "ccm": corrector.ccm.tolist(),
            "delta_e": report,
        }

    async def _capture_corrected(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Trigger a full-resolution still on the source camera (via its ``capture``
        DoCommand), color-correct it, and return it as base64 JPEG. ``opts`` is
        forwarded to the source capture (e.g. ``{"af": true}``).
        """
        source_resp = await self.camera.do_command({"capture": opts}, timeout=timeout)
        capture = source_resp.get("capture", source_resp)
        if not isinstance(capture, Mapping) or not capture.get("image_base64"):
            raise ValueError(
                "source camera `capture` did not return an `image_base64` field"
            )

        img_rgb = _base64_to_rgb(capture["image_base64"])
        corrected = self.corrector.apply_to_rgb(img_rgb)
        return {
            "image_base64": _rgb_to_base64_jpeg(corrected),
            "mime_type": CameraMimeType.JPEG.value,
            "path": capture.get("path"),
        }

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return await self.camera.get_geometries(extra=extra, timeout=timeout)
