"""Tests for the PTP model's pure helpers and the USB auto-recovery logic
(no camera hardware needed)."""

import asyncio

import pytest

from models.ptp import (
    PTP,
    PTPSession,
    _device_gone,
    _is_image,
    _mime_for,
    _retry_once_on_device_gone,
)

gp = pytest.importorskip("gphoto2")


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


# ---------------------------------------------------------------------------
# USB device-gone detection / auto-recovery
# ---------------------------------------------------------------------------

def test_device_gone_codes():
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO_USB_FIND))   # -52
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO_USB_CLAIM))  # -53
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO))            # -7
    # A generic capture failure (e.g. autofocus) is NOT a vanished device.
    assert not _device_gone(gp.GPhoto2Error(-1))
    assert not _device_gone(RuntimeError("not a gphoto error"))


class _FlakySession:
    """Stand-in for PTPSession: fails with a device-gone error until
    ``reconnect`` is called, mimicking a camera that slept and woke up."""

    def __init__(self, error_code=None, recoverable=True):
        self.error_code = error_code
        self.recoverable = recoverable
        self.calls = 0
        self.reconnects = 0

    def reconnect(self):
        self.reconnects += 1
        if self.recoverable:
            self.error_code = None

    @_retry_once_on_device_gone
    def op(self):
        self.calls += 1
        if self.error_code is not None:
            raise gp.GPhoto2Error(self.error_code)
        return "ok"


def test_retry_recovers_after_reconnect():
    session = _FlakySession(error_code=gp.GP_ERROR_IO_USB_FIND)
    assert session.op() == "ok"
    assert session.reconnects == 1
    assert session.calls == 2


def test_retry_gives_up_after_one_attempt():
    session = _FlakySession(error_code=gp.GP_ERROR_IO_USB_FIND, recoverable=False)
    with pytest.raises(gp.GPhoto2Error):
        session.op()
    assert session.reconnects == 1
    assert session.calls == 2  # original + exactly one retry, no loop


def test_retry_does_not_touch_other_errors():
    session = _FlakySession(error_code=-1)  # generic failure, not device-gone
    with pytest.raises(gp.GPhoto2Error):
        session.op()
    assert session.reconnects == 0
    assert session.calls == 1


def test_no_retry_when_healthy():
    session = _FlakySession()
    assert session.op() == "ok"
    assert session.calls == 1
    assert session.reconnects == 0


# ---------------------------------------------------------------------------
# capture(): retries the trigger once, and never re-fires a started capture
# ---------------------------------------------------------------------------

class _FakeCam:
    """Fake libgphoto2 Camera covering the calls capture() makes."""

    def __init__(self, trigger_errors):
        self.trigger_errors = list(trigger_errors)
        self.triggers = 0

    def trigger_capture(self):
        self.triggers += 1
        if self.trigger_errors:
            raise gp.GPhoto2Error(self.trigger_errors.pop(0))

    def wait_for_event(self, _timeout_ms):
        class _Added:
            folder = "/store/DCIM/100CANON"
            name = "IMG_0001.CR3"

        return gp.GP_EVENT_FILE_ADDED, _Added()


def _session_with_cam(cam):
    session = PTPSession.__new__(PTPSession)  # skip __init__'s gp probe
    session._camera = cam
    session.model_name = "Fake"
    session.port_path = "usb:000,000"
    session.reconnected = False

    def reconnect():
        session.reconnected = True

    session.reconnect = reconnect
    return session


def test_capture_retries_trigger_after_device_gone():
    cam = _FakeCam(trigger_errors=[gp.GP_ERROR_IO_USB_FIND])
    session = _session_with_cam(cam)
    path = session.capture(settle=0.1)
    assert path == "/store/DCIM/100CANON/IMG_0001.CR3"
    assert session.reconnected
    assert cam.triggers == 2


def test_capture_device_gone_twice_raises_usb_message():
    cam = _FakeCam(trigger_errors=[gp.GP_ERROR_IO_USB_FIND, gp.GP_ERROR_IO_USB_FIND])
    session = _session_with_cam(cam)
    with pytest.raises(RuntimeError, match="not reachable on USB"):
        session.capture(settle=0.1)
    assert cam.triggers == 2  # exactly one retry


def test_capture_generic_error_keeps_autofocus_hint_and_no_retry():
    cam = _FakeCam(trigger_errors=[-1])
    session = _session_with_cam(cam)
    with pytest.raises(RuntimeError, match="Check autofocus"):
        session.capture(settle=0.1)
    assert not session.reconnected
    assert cam.triggers == 1


# ---------------------------------------------------------------------------
# `trigger` DoCommand: fires the shutter, returns the on-camera path, and
# never downloads (the deferred-pipeline handoff)
# ---------------------------------------------------------------------------

class _TriggerOnlySession:
    """Fake PTPSession that fails loudly if anything tries to download."""

    def __init__(self):
        self.captures = 0

    def capture(self, settle):
        self.captures += 1
        return "/store/DCIM/100CANON/IMG_0042.CR3"

    def read_file(self, path):
        raise AssertionError("trigger must not download the file")


def _ptp_component(session):
    ptp = PTP("test-ptp")
    ptp._session = session
    ptp._lock = asyncio.Lock()
    ptp._capture_settle = 0.0
    ptp._download_dir = None
    ptp._delete_after_download = False
    ptp._downloaded = set()
    return ptp


def test_trigger_returns_camera_path_without_download():
    session = _TriggerOnlySession()
    ptp = _ptp_component(session)

    resp = asyncio.run(ptp.do_command({"trigger": {}}))

    out = resp["trigger"]
    assert out["path"] == "/store/DCIM/100CANON/IMG_0042.CR3"
    assert out["name"] == "IMG_0042.CR3"
    assert isinstance(out["mime_type"], str)
    assert "saved_to" not in out  # nothing was downloaded or written
    assert session.captures == 1
