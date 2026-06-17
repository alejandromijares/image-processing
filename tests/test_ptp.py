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


# ---------------------------------------------------------------------------
# cleanup(): clears the local download_dir without touching the camera card
# ---------------------------------------------------------------------------

def test_cleanup_removes_files_in_download_dir(tmp_path):
    (tmp_path / "IMG_0001.CR3").write_bytes(b"a")
    (tmp_path / "IMG_0002.JPG").write_bytes(b"b")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.JPG").write_bytes(b"c")  # subdirectory is left alone

    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = str(tmp_path)

    resp = asyncio.run(ptp.do_command({"cleanup": {}}))

    out = resp["cleanup"]
    assert out["count"] == 2
    assert set(out["removed"]) == {"IMG_0001.CR3", "IMG_0002.JPG"}
    assert out["dry_run"] is False
    assert not (tmp_path / "IMG_0001.CR3").exists()
    assert not (tmp_path / "IMG_0002.JPG").exists()
    assert (sub / "nested.JPG").exists()  # subdir contents untouched


def test_cleanup_dry_run_reports_without_deleting(tmp_path):
    (tmp_path / "IMG_0001.CR3").write_bytes(b"a")

    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = str(tmp_path)

    resp = asyncio.run(ptp.do_command({"cleanup": {"dry_run": True}}))

    out = resp["cleanup"]
    assert out["count"] == 1
    assert out["removed"] == ["IMG_0001.CR3"]
    assert out["dry_run"] is True
    assert (tmp_path / "IMG_0001.CR3").exists()  # nothing actually deleted


def test_cleanup_requires_download_dir():
    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = None

    with pytest.raises(ValueError, match="download_dir"):
        asyncio.run(ptp.do_command({"cleanup": {}}))


# ---------------------------------------------------------------------------
# list_widgets(): walks the camera's config tree for focus discovery, with an
# opt-in live-view toggle that exposes the step-drive widgets
# ---------------------------------------------------------------------------

class _FakeWidget:
    """Stand-in for a libgphoto2 CameraWidget covering the accessors the walk
    uses. ``get_value`` raises when ``raises=True`` to exercise skip-on-error."""

    def __init__(
        self, name, wtype, label=None, value=None, readonly=False,
        choices=None, rng=None, children=None, raises=False,
    ):
        self._name = name
        self._type = wtype
        self._label = name if label is None else label
        self._value = value
        self._readonly = readonly
        self._choices = choices
        self._range = rng
        self._children = children or []
        self._raises = raises
        self.set_values = []  # history of set_value() calls, for assertions

    def get_type(self):
        return self._type

    def get_name(self):
        return self._name

    def get_label(self):
        return self._label

    def get_readonly(self):
        return self._readonly

    def get_value(self):
        if self._raises:
            raise gp.GPhoto2Error(-1)
        return self._value

    def set_value(self, v):
        self._value = v
        self.set_values.append(v)

    def count_choices(self):
        return len(self._choices or [])

    def get_choice(self, i):
        return self._choices[i]

    def get_range(self):
        return self._range

    def get_children(self):
        return list(self._children)

    def get_child_by_name(self, name):
        # libgphoto2 searches descendants, not just direct children.
        found = self._find(name)
        if found is None:
            raise gp.GPhoto2Error(gp.GP_ERROR)
        return found

    def _find(self, name):
        if self._name == name:
            return self
        for child in self._children:
            hit = child._find(name)
            if hit is not None:
                return hit
        return None


class _FakeCamForConfig:
    """Fake Camera that hands out a (possibly evolving) config tree.

    ``get_config()`` returns successive trees by call index (clamped to the
    last), so a test can model the tree changing once live view turns on.
    ``set_config`` is recorded, and can be made to fail on chosen call numbers
    to exercise the best-effort restore.
    """

    def __init__(self, trees, fail_set_config=()):
        self._trees = trees
        self._fail_set_config = set(fail_set_config)
        self.get_config_calls = 0
        self.set_config_calls = 0

    def get_config(self):
        idx = min(self.get_config_calls, len(self._trees) - 1)
        self.get_config_calls += 1
        return self._trees[idx]

    def set_config(self, config):
        self.set_config_calls += 1
        if self.set_config_calls in self._fail_set_config:
            raise gp.GPhoto2Error(gp.GP_ERROR_IO_USB_FIND)


def _radio(name, choices, value):
    return _FakeWidget(name, gp.GP_WIDGET_RADIO, choices=choices, value=value)


def _root(children):
    return _FakeWidget("main", gp.GP_WIDGET_WINDOW, children=children)


def test_list_widgets_reports_all_widgets_with_focus_flag():
    tree = _root([
        _FakeWidget("capture", gp.GP_WIDGET_SECTION, children=[
            _radio("manualfocusdrive", ["None", "Near 1", "Far 1"], "None"),
            _radio("focusmode", ["One Shot", "Manual"], "Manual"),
            _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0),
            _radio("eosremoterelease", ["None", "Press Half", "Press Full"], "None"),
            _radio("iso", ["100", "200"], "100"),
            _FakeWidget("lightmeter", gp.GP_WIDGET_RANGE, value=0.0, rng=(-3.0, 3.0, 0.5)),
        ]),
    ])
    session = _session_with_cam(_FakeCamForConfig([tree]))

    widgets = session.list_widgets()
    by_name = {w["name"]: w for w in widgets}

    # Every leaf is reported - sections/windows are not, but iso is.
    assert set(by_name) == {
        "manualfocusdrive", "focusmode", "eosviewfinder",
        "eosremoterelease", "iso", "lightmeter",
    }
    # focus_relevant flags: focus widgets + name-based extras True, iso False.
    assert by_name["manualfocusdrive"]["focus_relevant"] is True
    assert by_name["focusmode"]["focus_relevant"] is True
    assert by_name["eosviewfinder"]["focus_relevant"] is True
    assert by_name["eosremoterelease"]["focus_relevant"] is True
    assert by_name["iso"]["focus_relevant"] is False
    # Choices and range are surfaced for calibration.
    assert by_name["manualfocusdrive"]["choices"] == ["None", "Near 1", "Far 1"]
    assert by_name["manualfocusdrive"]["type"] == "radio"
    assert by_name["lightmeter"]["range"] == {"min": -3.0, "max": 3.0, "step": 0.5}


def test_list_widgets_skips_unreadable_widget():
    tree = _root([
        _radio("focusmode", ["One Shot", "Manual"], "Manual"),
        _FakeWidget("brokenfocus", gp.GP_WIDGET_TEXT, raises=True),
    ])
    session = _session_with_cam(_FakeCamForConfig([tree]))

    widgets = session.list_widgets()
    names = {w["name"] for w in widgets}

    assert names == {"focusmode"}  # the raising widget is skipped, not fatal


def test_list_widgets_live_view_toggles_and_restores(monkeypatch):
    monkeypatch.setattr("models.ptp._LIVE_VIEW_SETTLE_SEC", 0.0)
    # The viewfinder widget is shared across trees so its set_value history
    # captures both the toggle-on and the restore.
    vf = _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0)
    before = _root([vf, _radio("focusmode", ["Manual"], "Manual")])
    after = _root([
        vf,
        _radio("focusmode", ["Manual"], "Manual"),
        _radio("manualfocusdrive", ["None", "Near 1"], "None"),
    ])
    cam = _FakeCamForConfig([before, after, after])
    session = _session_with_cam(cam)

    widgets = session.list_widgets(live_view=True)
    names = {w["name"] for w in widgets}

    # Re-fetch happened: manualfocusdrive only exists in the post-toggle tree.
    assert "manualfocusdrive" in names
    # Toggled on, then restored to the original value.
    assert vf.set_values == [1, 0]
    assert cam.set_config_calls == 2
    assert cam.get_config_calls == 3  # initial, re-fetch, restore


def test_list_widgets_restore_is_best_effort(monkeypatch):
    monkeypatch.setattr("models.ptp._LIVE_VIEW_SETTLE_SEC", 0.0)
    vf = _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0)
    before = _root([vf])
    after = _root([vf, _radio("manualfocusdrive", ["None"], "None")])
    # Fail the restore set_config (the 2nd one) with a device-gone error.
    cam = _FakeCamForConfig([before, after, after], fail_set_config={2})
    session = _session_with_cam(cam)

    widgets = session.list_widgets(live_view=True)
    names = {w["name"] for w in widgets}

    # The discovery result still comes back; the restore failure is swallowed
    # and does NOT propagate (which would otherwise trip the device-gone retry).
    assert "manualfocusdrive" in names
    assert session.reconnected is False


def test_do_command_list_widgets():
    class _ListWidgetsSession:
        def __init__(self, widgets):
            self._widgets = widgets
            self.calls = []

        def list_widgets(self, live_view=False):
            self.calls.append(live_view)
            return self._widgets

    session = _ListWidgetsSession([
        {"name": "manualfocusdrive", "focus_relevant": True, "value": (1, 2)},
        {"name": "iso", "focus_relevant": False, "value": "100"},
    ])
    ptp = _ptp_component(session)

    resp = asyncio.run(ptp.do_command({"list_widgets": {"live_view": True}}))
    out = resp["list_widgets"]

    assert out["count"] == 2
    assert out["focus_count"] == 1
    assert session.calls == [True]  # live_view forwarded
    # Non-primitive widget values are coerced to gRPC-safe types (tuple->list).
    by_name = {w["name"]: w for w in out["widgets"]}
    assert by_name["manualfocusdrive"]["value"] == [1, 2]
