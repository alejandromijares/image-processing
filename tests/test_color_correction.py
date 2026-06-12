"""Tests for the color-correction math: CCM fitting, application, sampling.

Only the Viam-decoupled pieces are exercised (ColorCorrector, PatchSampler,
_fit_ccm, _order_corners, detect_colorchecker plumbing) — no viam-server, no
camera hardware.
"""

import numpy as np
import pytest

from models.color_correction import (
    _oriented_chart_grid,
    REFERENCE_SRGB,
    ColorCorrector,
    PatchSampler,
    _fit_ccm,
    _order_corners,
)
from models.image_io import linear_to_srgb, srgb_to_linear


# ---------------------------------------------------------------------------
# _fit_ccm
# ---------------------------------------------------------------------------

def test_fit_ccm_recovers_known_matrix():
    """If reference = measured @ A.T exactly, the fit must recover A."""
    rng = np.random.default_rng(42)
    measured = rng.uniform(0.05, 0.9, size=(24, 3)).astype(np.float32)
    a = np.array(
        [[1.2, -0.1, 0.05],
         [-0.08, 1.1, -0.02],
         [0.03, -0.15, 1.3]],
        dtype=np.float32,
    )
    reference = measured @ a.T
    ccm = _fit_ccm(measured, reference)
    assert np.allclose(ccm, a, atol=1e-4)


def test_fit_ccm_identity_when_measured_equals_reference():
    reference = srgb_to_linear(REFERENCE_SRGB)
    ccm = _fit_ccm(reference, reference)
    assert np.allclose(ccm, np.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# ColorCorrector
# ---------------------------------------------------------------------------

def test_corrector_rejects_bad_shape():
    with pytest.raises(ValueError, match="3x3"):
        ColorCorrector(np.eye(4))


def test_identity_is_noop_passthrough():
    corrector = ColorCorrector.identity()
    assert corrector.is_identity
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    # Identity returns the input object untouched (no gamma round-trip).
    assert corrector.apply_to_rgb(img) is img
    linear = img.astype(np.float32) / 255.0
    assert corrector.apply_to_linear(linear) is linear


def test_apply_to_linear_matches_manual_matmul():
    ccm = np.array(
        [[0.9, 0.1, 0.0],
         [0.0, 1.0, 0.0],
         [0.05, -0.05, 1.0]],
        dtype=np.float32,
    )
    corrector = ColorCorrector(ccm)
    linear = np.random.default_rng(1).uniform(0, 1, (4, 5, 3)).astype(np.float32)
    out = corrector.apply_to_linear(linear)
    expected = (linear.reshape(-1, 3) @ ccm.T).reshape(4, 5, 3)
    assert out.shape == linear.shape
    assert np.allclose(out, expected, atol=1e-6)


def test_apply_to_rgb_round_trips_through_linear():
    """A diagonal gain CCM must scale colors in *linear* light, not sRGB."""
    ccm = np.diag([0.5, 1.0, 1.0]).astype(np.float32)
    corrector = ColorCorrector(ccm)
    img = np.full((4, 4, 3), 188, dtype=np.uint8)
    out = corrector.apply_to_rgb(img)
    expected_r = linear_to_srgb(srgb_to_linear(np.float32(188 / 255.0)) * 0.5) * 255.0
    assert np.allclose(out[..., 0], expected_r, atol=1)
    assert np.allclose(out[..., 1:], 188, atol=1)


def test_apply_to_rgb_clips_to_uint8_range():
    corrector = ColorCorrector(np.diag([3.0, 3.0, 3.0]).astype(np.float32))
    img = np.full((2, 2, 3), 250, dtype=np.uint8)
    out = corrector.apply_to_rgb(img)
    assert out.dtype == np.uint8
    assert out.max() == 255


# ---------------------------------------------------------------------------
# Corner ordering (chart detection geometry)
# ---------------------------------------------------------------------------

def test_order_corners_handles_any_winding():
    tl, tr, br, bl = (10, 20), (200, 25), (205, 150), (8, 145)
    for perm in ([br, tl, bl, tr], [tr, br, tl, bl], [bl, tr, br, tl]):
        ordered = _order_corners(np.array(perm, dtype=np.float32))
        assert np.allclose(ordered, np.array([tl, tr, br, bl], dtype=np.float32))


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def _synthetic_chart(patch_px: int = 60) -> np.ndarray:
    """Render REFERENCE_SRGB as a borderless 4x6 grid filling the frame."""
    rows, cols = 4, 6
    chart8 = (REFERENCE_SRGB * 255.0).round().astype(np.uint8)
    img = np.zeros((rows * patch_px, cols * patch_px, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            img[r * patch_px:(r + 1) * patch_px, c * patch_px:(c + 1) * patch_px] = (
                chart8[r * cols + c]
            )
    return img


def test_sample_at_centers_reads_exact_patches():
    img = _synthetic_chart()
    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    measured = PatchSampler.sample_at_centers(img, centers)
    assert measured.shape == (24, 3)
    assert np.allclose(measured, srgb_to_linear(REFERENCE_SRGB), atol=0.005)


def test_sample_linear_at_centers_clamps_at_edges():
    linear = np.random.default_rng(2).uniform(0, 1, (32, 32, 3)).astype(np.float32)
    # Centers at the very corners must not produce empty slices or wrap around.
    samples = PatchSampler.sample_linear_at_centers(linear, [(0, 0), (31, 31)], radius=10)
    assert samples.shape == (2, 3)
    assert np.all(np.isfinite(samples))


def test_calibrate_from_rgb_on_perfect_chart_is_near_identity():
    """A frame-filling chart at exactly the reference colors needs ~no correction."""
    img = _synthetic_chart()
    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    corrector = ColorCorrector.calibrate_from_rgb(img, patch_centers=centers)
    assert np.allclose(corrector.ccm, np.eye(3), atol=0.02)


def test_calibrate_from_rgb_corrects_a_cast():
    """Calibrating on a green-tinted chart yields a CCM that undoes the tint."""
    tint = np.diag([0.8, 1.1, 0.9]).astype(np.float32)
    img = _synthetic_chart()
    tinted_linear = srgb_to_linear(img.astype(np.float32) / 255.0) @ tint.T
    tinted = (linear_to_srgb(tinted_linear) * 255.0).round().astype(np.uint8)

    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    corrector = ColorCorrector.calibrate_from_rgb(tinted, patch_centers=centers)
    corrected = corrector.apply_to_rgb(tinted)

    ref8 = (REFERENCE_SRGB * 255.0).round()
    sampled = np.array([corrected[y, x] for x, y in centers], dtype=np.float32)
    assert np.abs(sampled - ref8).mean() < 3.0


# ---------------------------------------------------------------------------
# Orientation-robust chart grid (_oriented_chart_grid)
# ---------------------------------------------------------------------------

def _grid_corners(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_oriented_grid_resolves_any_90_degree_rotation(k):
    """A chart rotated k*90deg in frame must still map patches in reference order."""
    img = np.ascontiguousarray(np.rot90(_synthetic_chart(), k))
    detection = _oriented_chart_grid(img, _grid_corners(img))
    assert detection is not None
    assert detection["orientation_score"] > 2.0

    centers = [(int(x), int(y)) for x, y in detection["centers"]]
    measured = PatchSampler.sample_at_centers(img, centers)
    assert np.allclose(measured, srgb_to_linear(REFERENCE_SRGB), atol=0.005)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_oriented_grid_neutral_boxes_land_on_grays(k):
    """The WB boxes must cover Neutral 8 / 6.5 regardless of chart rotation."""
    img = np.ascontiguousarray(np.rot90(_synthetic_chart(), k))
    h, w = img.shape[:2]
    detection = _oriented_chart_grid(img, _grid_corners(img))
    assert detection is not None
    for box, expected in zip(detection["neutral_boxes_norm"], ([200] * 3, [160] * 3)):
        x0, y0, x1, y1 = box
        region = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        assert region.size > 0
        assert np.allclose(np.median(region.reshape(-1, 3), axis=0), expected, atol=2)


def test_oriented_grid_survives_white_balance_cast():
    """A strong channel cast (the wrong-WB case) must not confuse orientation."""
    img = _synthetic_chart()
    cast_linear = srgb_to_linear(img.astype(np.float32) / 255.0) * [0.4, 1.0, 2.2]
    cast = (linear_to_srgb(np.clip(cast_linear, 0, 1)) * 255).round().astype(np.uint8)
    cast = np.ascontiguousarray(np.rot90(cast, 1))

    detection = _oriented_chart_grid(cast, _grid_corners(cast))
    assert detection is not None

    upright = _synthetic_chart()
    upright_detection = _oriented_chart_grid(upright, _grid_corners(upright))
    # Same patch (white) must land at the rotated position of the upright one.
    ux, uy = upright_detection["centers"][18]
    rx, ry = detection["centers"][18]
    # rot90(k=1) maps (x, y) -> (y, W-1-x) where W is the original width
    assert abs(rx - uy) < 1.5 and abs(ry - (upright.shape[1] - 1 - ux)) < 1.5


def test_oriented_grid_rejects_non_chart():
    """A quad over random noise has no orientation that matches the reference."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(240, 360, 3), dtype=np.uint8)
    assert _oriented_chart_grid(noise, _grid_corners(noise)) is None


# ---------------------------------------------------------------------------
# Deferred capture (`capture` with `defer` + `capture_result`): the pipelined
# flow for rigs that move between shots. Exercised against a fake source
# camera - no viam-server, no hardware.
# ---------------------------------------------------------------------------

import asyncio

from PIL import Image

from models.color_correction import ColorCorrection


class _FakeSource:
    """Fake PTP-style source camera: `trigger` hands back an on-camera path,
    `download` "saves" a file that already exists at `saved_path`."""

    def __init__(self, saved_path, supports_trigger=True, saves_to_disk=True):
        self.saved_path = saved_path
        self.supports_trigger = supports_trigger
        self.saves_to_disk = saves_to_disk
        self.commands = []

    async def do_command(self, command, *, timeout=None, **kwargs):
        self.commands.append(command)
        if "trigger" in command:
            if not self.supports_trigger:
                raise ValueError("no recognized command")
            return {"trigger": {"path": "/store/DCIM/IMG_0042.PNG",
                                "name": "IMG_0042.PNG"}}
        if "download" in command:
            return {"download": {
                "path": command["download"]["path"],
                "name": "IMG_0042.PNG",
                "saved_to": self.saved_path if self.saves_to_disk else None,
            }}
        if "capture" in command:
            return {"capture": {"saved_to": self.saved_path}}
        raise ValueError("no recognized command")


def _component(source, output_dir=None):
    cc = ColorCorrection("test-cc")
    cc.camera = source
    cc.corrector = ColorCorrector.identity()
    cc._white_balance = "camera"
    cc._output_formats = ["tiff16", "jpeg"]
    cc._output_dir = output_dir
    cc._jpeg_quality = 95
    cc._write_sidecar = False
    cc._part_id = None
    cc._data_client = None
    cc._pending_captures = {}
    cc._capture_seq = 0
    return cc


def _write_still(tmp_path):
    p = str(tmp_path / "IMG_0042.PNG")
    Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(p, format="PNG")
    return p


def test_deferred_capture_round_trip(tmp_path):
    source = _FakeSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))

    async def run():
        ticket = (await cc.do_command({"capture": {"defer": True}}))["capture"]
        assert ticket["status"] == "pending"
        assert ticket["camera_path"] == "/store/DCIM/IMG_0042.PNG"
        result = (await cc.do_command(
            {"capture_result": {"id": ticket["capture_id"], "wait_sec": 30}}
        ))["capture_result"]
        return ticket, result

    ticket, result = asyncio.run(run())
    assert result["status"] == "done"
    assert result["source_path"] == source.saved_path
    assert result["image_base64"]  # preview present
    # Deferred captures hand off the RAW only - no exports, no sidecar.
    assert "exports" not in result
    # The ticket is collected exactly once.
    with pytest.raises(ValueError, match="unknown capture id"):
        asyncio.run(cc._capture_result({"id": ticket["capture_id"]}))


def test_deferred_capture_requires_trigger_support():
    source = _FakeSource(saved_path=None, supports_trigger=False)
    cc = _component(source)
    with pytest.raises(RuntimeError, match="`trigger`"):
        asyncio.run(cc.do_command({"capture": {"defer": True}}))


def test_deferred_capture_surfaces_background_failure(tmp_path):
    """A source without a download_dir fails in the background task; the
    error must surface on collect, not vanish."""
    source = _FakeSource(saved_path=None, saves_to_disk=False)
    cc = _component(source, output_dir=str(tmp_path / "out"))

    async def run():
        ticket = (await cc.do_command({"capture": {"defer": True}}))["capture"]
        with pytest.raises(RuntimeError, match="download_dir"):
            await cc.do_command(
                {"capture_result": {"id": ticket["capture_id"], "wait_sec": 30}}
            )

    asyncio.run(run())


def test_preview_only_capture_skips_exports(tmp_path):
    """`output_formats: []` is the preview-only fast path: no files written,
    preview still returned, RAW path handed back for a later `develop`."""
    source = _FakeSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))

    resp = asyncio.run(cc.do_command({"capture": {"output_formats": []}}))
    out = resp["capture"]
    assert out["exports"] == {}
    assert out["image_base64"]
    assert out["source_path"] == source.saved_path
