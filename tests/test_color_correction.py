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
    _neutral_brightness_report,
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


def test_neutral_brightness_report_at_nominal_matches_reference():
    """Patches measured at exactly the reference colors read back the
    reference 0-255 values (e.g. Neutral 6.5 -> 160)."""
    report = _neutral_brightness_report(srgb_to_linear(REFERENCE_SRGB))
    assert set(report) == {
        "white_9_5", "neutral_8", "neutral_6_5", "neutral_5", "neutral_3_5", "black_2"
    }
    for patch in report.values():
        assert patch["measured"] == pytest.approx(patch["reference"], abs=0.5)
    assert report["neutral_6_5"]["reference"] == pytest.approx(160.0, abs=0.5)


def test_neutral_brightness_report_tracks_light_power():
    """A one-stop-under chart reads darker than reference on every patch -
    the readout the user watches while dialing flash power."""
    under = srgb_to_linear(REFERENCE_SRGB) * 0.5
    report = _neutral_brightness_report(under)
    for patch in report.values():
        assert patch["measured"] < patch["reference"]
    # Half the linear light on Neutral 6.5 (0.353 linear) lands around sRGB 117.
    assert 110 < report["neutral_6_5"]["measured"] < 125


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
    cc._exposure_stops = 0.0
    cc._output_formats = ["tiff16", "jpeg"]
    cc._output_dir = output_dir
    cc._jpeg_quality = 95
    cc._write_sidecar = False
    cc._part_id = None
    cc._data_client = None
    cc._delete_after_upload = False
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


def test_configured_exposure_stops_flows_into_develop(tmp_path, monkeypatch):
    """The `exposure_stops` config default reaches the raw decode when a call
    doesn't override it (the digital counterpart to flash power), and a per-call
    value still wins."""
    import models.color_correction as cc_mod

    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    cc._exposure_stops = 0.87

    seen = {}
    real = cc_mod.load_linear_rgb

    def spy(path, **kwargs):
        seen["exposure_stops"] = kwargs.get("exposure_stops")
        return real(path, **kwargs)

    monkeypatch.setattr(cc_mod, "load_linear_rgb", spy)
    p = _write_still(tmp_path)

    asyncio.run(cc.do_command({"develop": {"path": p}}))
    assert seen["exposure_stops"] == 0.87  # config default applied

    asyncio.run(cc.do_command({"develop": {"path": p, "exposure_stops": 0.0}}))
    assert seen["exposure_stops"] == 0.0  # per-call override wins


# ---------------------------------------------------------------------------
# Chunked file upload: app.viam.com caps gRPC messages at 32 MiB, so `upload`
# must stream files in pieces rather than one message per file (which is what
# the SDK's `file_upload` does, silently dropping every CR3/TIFF). Exercised
# against a fake FileUpload stream - no cloud.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from viam.proto.app.datasync import FileUploadResponse

from models.color_correction import UPLOAD_CHUNK_BYTES


class _FakeUploadStream:
    """Records every (request, end) pair sent over the FileUpload stream."""

    def __init__(self, sent):
        self.sent = sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send_message(self, msg, end=False):
        self.sent.append((msg, end))

    async def recv_message(self):
        return FileUploadResponse(binary_data_id="fake-binary-id")


class _FakeDataClient:
    def __init__(self):
        self.sent = []
        self._metadata = {"authorization": "Bearer fake"}
        self._data_sync_client = SimpleNamespace(
            FileUpload=SimpleNamespace(
                open=lambda metadata=None: _FakeUploadStream(self.sent)
            )
        )


def test_chunked_upload_splits_large_file(tmp_path):
    """A file larger than one chunk goes out as metadata + several FileData
    messages, each under the cap, reassembling to the original bytes."""
    payload = bytes(range(256)) * ((2 * UPLOAD_CHUNK_BYTES + 1234) // 256 + 1)
    path = tmp_path / "big.tif"
    path.write_bytes(payload)

    client = _FakeDataClient()
    binary_id = asyncio.run(
        ColorCorrection._file_upload_chunked(
            client, str(path),
            part_id="part-1", component_name="cc", tags=["sku:123"],
        )
    )

    assert binary_id == "fake-binary-id"
    meta_msg, meta_end = client.sent[0]
    assert meta_msg.metadata.file_name == "big.tif"
    assert meta_msg.metadata.file_extension == ".tif"
    assert list(meta_msg.metadata.tags) == ["sku:123"]
    assert meta_end is False

    chunks = [msg.file_contents.data for msg, _ in client.sent[1:]]
    assert len(chunks) > 1
    assert all(len(c) <= UPLOAD_CHUNK_BYTES for c in chunks)
    assert b"".join(chunks) == payload
    # Only the final message closes the stream.
    ends = [end for _, end in client.sent[1:]]
    assert ends == [False] * (len(ends) - 1) + [True]


def test_chunked_upload_empty_file_sends_one_chunk(tmp_path):
    """An empty file still sends one (empty) FileData message, matching the
    SDK's behavior, so the stream is closed properly."""
    path = tmp_path / "empty.json"
    path.write_bytes(b"")

    client = _FakeDataClient()
    asyncio.run(
        ColorCorrection._file_upload_chunked(
            client, str(path), part_id="p", component_name="cc", tags=None,
        )
    )

    assert len(client.sent) == 2  # metadata + one empty chunk
    chunk_msg, chunk_end = client.sent[1]
    assert chunk_msg.file_contents.data == b""
    assert chunk_end is True


# ---------------------------------------------------------------------------
# Local-disk cleanup: nothing else in the pipeline ever deletes a local file,
# so the download dir grows by every frame shot. `delete` removes skipped
# captures (guarded to output_dir); `delete_after_upload` removes each file
# once it is confirmed in the cloud.
# ---------------------------------------------------------------------------


def _uploader_component(tmp_path, monkeypatch, fail_paths=()):
    """Component with upload wired to a fake cloud; uploads of `fail_paths` fail."""
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    cc._part_id = "part-1"

    async def fake_get_data_client():
        return _FakeDataClient()

    async def fake_upload_chunked(client, path, **kwargs):
        if path in fail_paths:
            raise RuntimeError("simulated upload failure")
        return "fake-binary-id"

    monkeypatch.setattr(cc, "_get_data_client", fake_get_data_client)
    monkeypatch.setattr(ColorCorrection, "_file_upload_chunked",
                        staticmethod(fake_upload_chunked))
    return cc


def test_upload_keeps_files_by_default(tmp_path, monkeypatch):
    path = tmp_path / "a.CR3"
    path.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch)

    out = asyncio.run(cc._upload({"paths": [str(path)]}))
    assert out["uploaded"] == [str(path)]
    assert out["deleted"] == []
    assert path.exists()


def test_delete_after_upload_removes_only_successful(tmp_path, monkeypatch):
    """Uploaded files are deleted; a failed upload keeps its file for retry."""
    ok = tmp_path / "a.CR3"
    bad = tmp_path / "b.CR3"
    ok.write_bytes(b"raw")
    bad.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch, fail_paths=(str(bad),))
    cc._delete_after_upload = True

    out = asyncio.run(cc._upload({"paths": [str(ok), str(bad)]}))
    assert out["uploaded"] == [str(ok)]
    assert out["deleted"] == [str(ok)]
    assert [f["path"] for f in out["failed"]] == [str(bad)]
    assert not ok.exists()
    assert bad.exists()


def test_delete_after_upload_command_overrides_config(tmp_path, monkeypatch):
    path = tmp_path / "a.CR3"
    path.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch)  # config default: keep

    out = asyncio.run(
        cc._upload({"paths": [str(path)], "delete_after_upload": True})
    )
    assert out["deleted"] == [str(path)]
    assert not path.exists()


def test_delete_removes_files_inside_output_dir(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    keep = tmp_path / "keep.CR3"
    drop = tmp_path / "drop.CR3"
    keep.write_bytes(b"raw")
    drop.write_bytes(b"raw")

    out = asyncio.run(cc.do_command({"delete": {"paths": [str(drop)]}}))["delete"]
    assert out["deleted"] == [str(drop)]
    assert out["count"] == 1
    assert not drop.exists()
    assert keep.exists()


def test_delete_missing_file_is_idempotent(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    gone = str(tmp_path / "already-gone.CR3")

    out = asyncio.run(cc.do_command({"delete": {"paths": [gone]}}))["delete"]
    assert out["deleted"] == []
    assert out["missing"] == [gone]
    assert out["failed"] == []


def test_delete_refuses_paths_outside_output_dir(tmp_path):
    """Absolute paths, `..` traversal, and symlinks out of output_dir are all
    refused - the command must not be able to reach the rest of the host."""
    images = tmp_path / "images"
    images.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"important")
    link = images / "sneaky.CR3"
    link.symlink_to(outside)
    cc = _component(_FakeSource(None), output_dir=str(images))

    out = asyncio.run(cc.do_command({"delete": {"paths": [
        str(outside),
        str(images / ".." / "secret.txt"),
        str(link),
    ]}}))["delete"]

    assert out["deleted"] == []
    assert len(out["failed"]) == 3
    assert all("outside output_dir" in f["error"] for f in out["failed"])
    assert outside.exists()


def test_delete_requires_output_dir(tmp_path):
    cc = _component(_FakeSource(None), output_dir=None)
    with pytest.raises(ValueError, match="output_dir"):
        asyncio.run(cc.do_command({"delete": {"paths": ["/anything"]}}))


def test_delete_requires_paths(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="paths"):
        asyncio.run(cc.do_command({"delete": {}}))
