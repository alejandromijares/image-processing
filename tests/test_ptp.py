"""Tests for the PTP model's pure helpers (no camera hardware needed)."""

from models.ptp import _is_image, _mime_for


def test_is_image_accepts_stills_and_raws():
    assert _is_image("IMG_0042.JPG")
    assert _is_image("IMG_0042.cr3")
    assert _is_image("DSC_0001.NEF")
    assert not _is_image("MVI_0042.MP4")
    assert not _is_image("IMG_0042.JPG.tmp")


def test_mime_for_known_and_opaque_types():
    assert _mime_for("IMG_0042.JPG") == "image/jpeg"
    assert _mime_for("shot.png") == "image/png"
    # RAW is opaque bytes-on-the-wire, not a previewable image.
    assert _mime_for("IMG_0042.CR3") == "application/octet-stream"
