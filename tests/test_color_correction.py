"""Tests for the color-correction math: CCM fitting, application, sampling.

Only the Viam-decoupled pieces are exercised (ColorCorrector, PatchSampler,
_fit_ccm, _order_corners, detect_colorchecker plumbing) — no viam-server, no
camera hardware.
"""

import numpy as np
import pytest

from models.color_correction import (
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
