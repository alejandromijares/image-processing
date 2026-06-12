"""Tests for image_io: decode, transfer functions, WB option parsing, exports."""

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from models.image_io import (
    EXPORT_FORMATS,
    _encode_srgb,
    _rawpy_wb_kwargs,
    export_renditions,
    is_raw,
    linear_to_jpeg_base64,
    linear_to_srgb,
    load_linear_rgb,
    srgb_to_linear,
)


# ---------------------------------------------------------------------------
# sRGB transfer functions
# ---------------------------------------------------------------------------

def test_srgb_round_trip():
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_srgb_known_values():
    # 0 and 1 are fixed points; mid-grey 0.5 sRGB is ~0.2140 linear.
    assert srgb_to_linear(np.float32(0.0)) == 0.0
    assert np.isclose(srgb_to_linear(np.float32(1.0)), 1.0, atol=1e-6)
    assert np.isclose(srgb_to_linear(np.float32(0.5)), 0.21404, atol=1e-4)


def test_linear_to_srgb_clips_out_of_range():
    out = linear_to_srgb(np.array([-0.5, 2.0], dtype=np.float32))
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# RAW detection / white-balance option parsing
# ---------------------------------------------------------------------------

def test_is_raw():
    assert is_raw("/photos/IMG_0042.CR3")
    assert is_raw("shot.nef")
    assert not is_raw("shot.jpg")
    assert not is_raw("shot.tiff")


@pytest.mark.parametrize(
    "option,expected",
    [
        ("camera", {"use_camera_wb": True}),
        ("as-shot", {"use_camera_wb": True}),
        ("auto", {"use_auto_wb": True}),
        ("daylight", {}),
        ("none", {}),
        (None, {}),
        ([2.0, 1.0, 1.5, 1.0], {"user_wb": [2.0, 1.0, 1.5, 1.0]}),
    ],
)
def test_rawpy_wb_kwargs(option, expected):
    assert _rawpy_wb_kwargs(option) == expected


def test_rawpy_wb_kwargs_rejects_bad_input():
    with pytest.raises(ValueError):
        _rawpy_wb_kwargs("tungsten")
    with pytest.raises(ValueError):
        _rawpy_wb_kwargs([1.0, 2.0, 3.0])  # needs 4 multipliers


# ---------------------------------------------------------------------------
# load_linear_rgb (non-RAW path — regression for the stranded PIL fallback)
# ---------------------------------------------------------------------------

def _write_solid(tmp_path, name, value, fmt):
    p = str(tmp_path / name)
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(p, format=fmt)
    return p


def test_load_linear_rgb_png(tmp_path):
    """A solid sRGB PNG decodes to its exact linearized value."""
    p = _write_solid(tmp_path, "grey.png", 188, "PNG")
    out = load_linear_rgb(p)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (8, 8, 3)
    expected = srgb_to_linear(np.float32(188 / 255.0))
    assert np.allclose(out, expected, atol=1e-4)


def test_load_linear_rgb_jpeg(tmp_path):
    p = _write_solid(tmp_path, "grey.jpg", 128, "JPEG")
    out = load_linear_rgb(p)
    assert out is not None  # the original bug: non-RAW returned None
    assert out.shape == (8, 8, 3)
    # JPEG is lossy; just require the value to be near the encoded grey.
    expected = srgb_to_linear(np.float32(128 / 255.0))
    assert np.allclose(out, expected, atol=0.02)


def test_load_linear_rgb_missing_file():
    with pytest.raises(FileNotFoundError):
        load_linear_rgb("/nonexistent/IMG_0001.CR3")


# ---------------------------------------------------------------------------
# Encode / export
# ---------------------------------------------------------------------------

def test_encode_srgb_bit_depths():
    linear = np.full((2, 2, 3), 1.0, dtype=np.float32)
    out16 = _encode_srgb(linear, 16)
    out8 = _encode_srgb(linear, 8)
    assert out16.dtype == np.uint16 and out16.max() == 65535
    assert out8.dtype == np.uint8 and out8.max() == 255


def test_export_renditions_writes_all_formats(tmp_path):
    linear = np.full((16, 16, 3), srgb_to_linear(np.float32(0.5)), dtype=np.float32)
    written = export_renditions(linear, str(tmp_path), "shot", list(EXPORT_FORMATS))

    assert set(written) == set(EXPORT_FORMATS)
    for fmt, path in written.items():
        assert path.endswith(str(EXPORT_FORMATS[fmt]["suffix"]))

    # 8-bit formats round-trip through PIL at the encoded grey value.
    for fmt in ("tiff8", "png8"):
        arr = np.array(Image.open(written[fmt]))
        assert arr.dtype == np.uint8
        assert np.allclose(arr, 128, atol=1)

    # 16-bit TIFF master keeps full precision.
    import tifffile

    arr16 = tifffile.imread(written["tiff16"])
    assert arr16.dtype == np.uint16
    assert np.allclose(arr16, round(0.5 * 65535), atol=130)  # within ~0.2%

    # 16-bit PNG (written via cv2 in BGR) reads back uint16 and channel-symmetric.
    import cv2

    png16 = cv2.imread(written["png16"], cv2.IMREAD_UNCHANGED)
    assert png16.dtype == np.uint16


def test_export_renditions_rejects_unknown_format(tmp_path):
    linear = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown export format"):
        export_renditions(linear, str(tmp_path), "shot", ["webp"])


def test_16bit_suffixes_do_not_collide_with_8bit():
    assert EXPORT_FORMATS["tiff16"]["suffix"] != EXPORT_FORMATS["tiff8"]["suffix"]
    assert EXPORT_FORMATS["png16"]["suffix"] != EXPORT_FORMATS["png8"]["suffix"]


def test_linear_to_jpeg_base64_downsizes():
    linear = np.full((2048, 1024, 3), 0.2, dtype=np.float32)
    b64 = linear_to_jpeg_base64(linear, max_dim=256)
    img = Image.open(BytesIO(base64.b64decode(b64)))
    assert img.format == "JPEG"
    assert max(img.size) <= 256
    # Aspect ratio preserved (2:1).
    assert img.size[1] == 256 and img.size[0] == 128


def test_linear_to_jpeg_base64_stride_keeps_values():
    """The pre-gamma stride downsample is a pure speed move - a solid frame
    must come out at the same encoded value (and size) as the slow path."""
    value = srgb_to_linear(np.float32(150 / 255.0))
    linear = np.full((1200, 900, 3), value, dtype=np.float32)
    b64 = linear_to_jpeg_base64(linear, max_dim=128)
    img = Image.open(BytesIO(base64.b64decode(b64)))
    assert img.size == (96, 128)  # 4:3 preserved through stride + thumbnail
    arr = np.array(img.convert("RGB"))
    assert np.allclose(arr, 150, atol=2)  # JPEG-lossy tolerance


def test_linear_to_jpeg_base64_no_stride_on_small_input():
    """Inputs already near the target size must not be stride-sampled away."""
    linear = np.full((300, 200, 3), 0.5, dtype=np.float32)
    b64 = linear_to_jpeg_base64(linear, max_dim=256)
    img = Image.open(BytesIO(base64.b64decode(b64)))
    # stride must stay 1 here; only thumbnail() shrinks (300 -> 256 exactly).
    assert img.size[1] == 256 and 170 <= img.size[0] <= 171


# ---------------------------------------------------------------------------
# RAW decode options (rawpy faked - no real CR3 in unit tests)
# ---------------------------------------------------------------------------

class _FakeRawpy:
    """Stands in for the rawpy module: records postprocess kwargs."""

    class ColorSpace:
        sRGB = "srgb"

    def __init__(self):
        self.postprocess_kwargs = None

    def imread(self, path):
        fake = self

        class _Raw:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def postprocess(self, **kwargs):
                fake.postprocess_kwargs = kwargs
                return np.zeros((4, 4, 3), dtype=np.uint16)

        return _Raw()


def test_load_linear_rgb_half_size_passthrough(tmp_path, monkeypatch):
    import models.image_io as image_io

    fake = _FakeRawpy()
    monkeypatch.setattr(image_io, "rawpy", fake)
    raw_path = tmp_path / "IMG_0042.CR3"
    raw_path.write_bytes(b"not really a raw")

    load_linear_rgb(str(raw_path), half_size=True)
    assert fake.postprocess_kwargs["half_size"] is True

    load_linear_rgb(str(raw_path))
    assert "half_size" not in fake.postprocess_kwargs
