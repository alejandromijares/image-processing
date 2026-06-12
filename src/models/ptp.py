"""
ptp.py
------
A Viam camera component that talks directly to a still camera over USB (USB-C)
using PTP (Picture Transfer Protocol), via libgphoto2. Use it to pull images off
a Canon / Nikon / Sony / Fujifilm / etc. body without an SD-card reader.

Unlike the ``color-correction`` model, this one does **not** wrap another Viam
camera - it owns the USB connection itself. libgphoto2 only lets a single process
hold the camera at a time, so all access is serialized through an asyncio lock
and run in a thread executor (the gphoto2 calls are blocking).

Two ways to get images out:

1. Streaming path - ``get_images`` returns a live-view preview frame (a
   downsized JPEG from the camera's mirror-up live view). This is what the
   control tab shows. Not every body supports live view; if yours doesn't,
   ``get_images`` falls back to the most recent still on the card.

2. DoCommand path - the real PTP workflow:

       {"capture": {}}
           -> trip the shutter, download the resulting full-res still to
              `download_dir`, and return its metadata
              {"name", "path", "mime_type", "saved_to", "size"}. The bytes are
              not base64'd into the response - full-res stills (especially RAW)
              are too large for gRPC, so they move by file path (`saved_to`).
              Many bodies (notably Canon) write the still to the card themselves
              and report the capture as a benign libgphoto2 -1 without handing
              back the path; in that case we wait `capture_settle` seconds for
              the write to finish and download the newest file on the card.

       {"list_files": {}}
           -> enumerate the image files on the camera's storage.
              Options: {"new_only": true} to only list files not yet
              downloaded this session.

       {"download": {"path": "/store_00010001/DCIM/100CANON/IMG_042.JPG"}}
       {"download": {"latest": true}}
           -> download a file (or the newest one) to `download_dir` and return
              the same metadata shape as `capture`.

       {"download_all": {"new_only": true}}
           -> download every image (or only new ones) to `download_dir`,
              returning the list of saved paths. Avoids base64-ing a whole
              card back over gRPC.

       {"delete": {"path": "..."}}
           -> delete a file from the camera (only do this after a successful
              download; with `delete_after_download` it happens automatically).

       {"summary": {}}
           -> camera model, port, and the libgphoto2 capability summary.
"""

import asyncio
import functools
import os
import time
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

from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

# libgphoto2 is an optional system-backed dependency; importing it lazily lets
# the module load (and report a clean error) even where the wheel is missing.
try:
    import gphoto2 as gp  # type: ignore

    _GP_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    gp = None  # type: ignore
    _GP_IMPORT_ERROR = exc

LOGGER = getLogger(__name__)

# Extensions libgphoto2 reports that we treat as still images worth listing /
# downloading. JPEGs stream and preview fine; RAWs download but can't preview.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".heic")

_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".heic": "image/heic",
}


def _is_image(name: str) -> bool:
    return name.lower().endswith(_IMAGE_EXTS)


def _mime_for(name: str) -> str:
    _, ext = os.path.splitext(name.lower())
    # RAW and anything unknown is opaque to us; label it generically so callers
    # know it's bytes-on-the-wire, not a previewable JPEG.
    return _EXT_TO_MIME.get(ext, "application/octet-stream")


def _device_gone(exc: Exception) -> bool:
    """True if a libgphoto2 error means the camera vanished from USB.

    Bodies auto-power-off after a few idle minutes (and cables get pulled);
    from then on the session's handle is stale and every call fails with one of
    these codes until the connection is rebuilt.
    """
    if gp is None or not isinstance(exc, gp.GPhoto2Error):
        return False
    return getattr(exc, "code", None) in (
        gp.GP_ERROR_IO_USB_FIND,   # -52: device no longer on the port
        gp.GP_ERROR_IO_USB_CLAIM,  # -53: can't claim the interface
        gp.GP_ERROR_IO,            # -7:  I/O broke mid-conversation
    )


def _retry_once_on_device_gone(method):
    """Reconnect and retry a ``PTPSession`` method once if the camera vanished.

    Safe only for idempotent operations (reads, deletes): ``capture`` does its
    own recovery around the shutter trigger so a frame is never fired twice.
    The retry runs against a fresh connection; if the camera is still absent,
    ``reconnect`` raises the clear "no PTP camera detected" error instead.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            if not _device_gone(exc):
                raise
            LOGGER.warning(
                f"camera unreachable during {method.__name__} ({exc}); "
                "reconnecting and retrying once"
            )
            self.reconnect()
            return method(self, *args, **kwargs)
    return wrapper


class PTPSession:
    """
    Thin, *synchronous* wrapper around a libgphoto2 ``Camera`` connection.

    Decoupled from Viam and asyncio so the PTP logic can be unit-tested or
    scripted. Every method here blocks; the Viam model below runs them in a
    thread executor under a lock. Not thread-safe on its own - one caller at a
    time.
    """

    def __init__(self, port: Optional[str] = None, model_match: Optional[str] = None):
        if gp is None:
            raise RuntimeError(
                "python-gphoto2 is not available "
                f"({_GP_IMPORT_ERROR}); install `gphoto2` (which bundles "
                "libgphoto2) to use the PTP camera model"
            )
        self._port = port
        self._model_match = model_match
        self._camera = None  # type: ignore
        self.model_name: str = ""
        self.port_path: str = ""

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def autodetect() -> List[Tuple[str, str]]:
        """Return [(model, port), ...] for every PTP camera currently on USB."""
        if gp is None:
            return []
        return [(name, port) for name, port in gp.Camera.autodetect()]

    def open(self) -> None:
        """
        Initialise the USB connection. If a ``port`` or ``model_match`` was
        given, bind to that specific body; otherwise grab the first detected
        camera. Raises if none is connected.
        """
        cameras = self.autodetect()
        if not cameras:
            raise RuntimeError(
                "no PTP camera detected on USB - check the cable (use a data, "
                "not charge-only, USB-C cable), power the camera on, and make "
                "sure no other app (Photos, gphoto2, EOS Utility) holds it"
            )

        port = self._port
        model = self._model_match
        chosen = None
        for cam_model, cam_port in cameras:
            if port and cam_port != port:
                continue
            if model and model.lower() not in cam_model.lower():
                continue
            chosen = (cam_model, cam_port)
            break
        if chosen is None:
            detected = ", ".join(f"{m} @ {p}" for m, p in cameras)
            raise RuntimeError(
                f"no PTP camera matched (port={port!r}, model={model!r}); "
                f"detected: {detected}"
            )

        camera = gp.Camera()
        # Pin the connection to the chosen port so we don't race other bodies.
        port_info_list = gp.PortInfoList()
        port_info_list.load()
        idx = port_info_list.lookup_path(chosen[1])
        camera.set_port_info(port_info_list[idx])
        camera.init()

        self._camera = camera
        self.model_name = chosen[0]
        self.port_path = chosen[1]

        # Default capture to the memory card. Some bodies ship set to
        # "Internal RAM", which makes capture fail or fill up after a few
        # frames; the card is what we list/download from anyway.
        self._set_capture_target_card()

    def _set_capture_target_card(self) -> None:
        """Best-effort: point `capturetarget` at the memory card.

        Not all bodies expose this setting, and it's never fatal if missing,
        so any libgphoto2 error here is logged and swallowed.
        """
        cam = self._camera
        try:
            config = cam.get_config()
            target = config.get_child_by_name("capturetarget")
            for i in range(target.count_choices()):
                choice = str(target.get_choice(i))
                if "card" in choice.lower():
                    target.set_value(choice)
                    cam.set_config(config)
                    return
        except gp.GPhoto2Error as exc:  # body doesn't expose it; fine
            LOGGER.debug(f"could not set capturetarget to card: {exc}")

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.exit()
            except Exception:
                pass
            self._camera = None

    def reconnect(self) -> None:
        """Drop a stale connection and re-open from scratch (re-autodetect).

        Used after the camera vanishes from USB (auto power-off, replug): the
        old handle is unusable, and unlike ``refresh`` the device may have come
        back on a different port address, so a full ``open()`` is required.
        """
        self.close()
        self.open()

    @property
    def _cam(self):
        if self._camera is None:
            self.open()
        return self._camera

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @_retry_once_on_device_gone
    def summary(self) -> str:
        return str(self._cam.get_summary().text)

    def refresh(self) -> None:
        """Drop libgphoto2's cached filesystem so the next read re-scans the card.

        libgphoto2 reads the card's directory listing once at ``init()`` and
        caches it for the life of the connection. It does *not* reliably update
        that cache from PTP events, so frames shot - or files deleted - on the
        body while we're connected won't appear (or clear) until the cache is
        rebuilt. That's why a module restart "fixes" it: restart re-inits.

        ``gp_camera_exit()`` frees the cached filesystem; the next ``init()``
        re-reads it. We reuse the same ``Camera`` object and its port binding,
        so this is a USB re-handshake, not a full re-autodetect. If the
        re-handshake fails we fall back to a clean reopen.
        """
        cam = self._camera
        if cam is None:
            return
        try:
            cam.exit()
            cam.init()
        except gp.GPhoto2Error as exc:
            LOGGER.debug(f"refresh re-handshake failed ({exc}); reopening")
            self.close()
            self.open()

    @_retry_once_on_device_gone
    def list_image_files(self, folder: str = "/") -> List[str]:
        """Recursively list image file paths on the camera's storage."""
        self.refresh()
        return self._walk_image_files(folder)

    def _walk_image_files(self, folder: str) -> List[str]:
        cam = self._cam
        files: List[str] = []
        for name, _ in cam.folder_list_files(folder):
            if _is_image(name):
                files.append(folder.rstrip("/") + "/" + name)
        for name, _ in cam.folder_list_folders(folder):
            sub = folder.rstrip("/") + "/" + name
            files.extend(self._walk_image_files(sub))
        return files

    def latest_image_file(self) -> Optional[str]:
        """Path of the most recently captured image, or None if the card is empty."""
        files = self.list_image_files()
        # File paths sort lexically with capture order on every body I've seen
        # (DCIM/100CANON/IMG_0001 ...). Good enough to pick "newest".
        return sorted(files)[-1] if files else None

    @_retry_once_on_device_gone
    def read_file(self, path: str) -> bytes:
        """Download a single file's bytes by full camera path."""
        folder, name = os.path.split(path)
        camera_file = self._cam.file_get(folder, name, gp.GP_FILE_TYPE_NORMAL)
        return bytes(camera_file.get_data_and_size())

    @_retry_once_on_device_gone
    def preview(self) -> bytes:
        """Grab a single live-view preview frame (JPEG bytes)."""
        camera_file = self._cam.capture_preview()
        return bytes(camera_file.get_data_and_size())

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def capture(self, settle: float = 2.0) -> str:
        """Fire the shutter and return the new file's full camera path.

        We use ``trigger_capture()`` rather than ``capture(GP_CAPTURE_IMAGE)``:
        the latter blocks up to ~60s waiting for a FILE_ADDED event that many
        bodies (notably Canon writing to the card) never send, then raises a
        generic -1. ``trigger_capture()`` trips the shutter and returns at once.

        After firing we drain the event queue for up to ``settle`` seconds. That
        gives the body time to finish writing and lets libgphoto2 process events:
        a body that *does* report the new file hands us its path directly (fast
        path), and one that writes to the card silently falls through to the
        newest file on the card.
        """
        def _capture_error(exc: Exception) -> RuntimeError:
            if _device_gone(exc):
                return RuntimeError(
                    f"camera not reachable on USB (libgphoto2: {exc}); it has "
                    "likely auto-powered off to sleep or been unplugged - wake "
                    "it (half-press the shutter), check the cable, and consider "
                    "disabling auto power-off for tethered work."
                )
            return RuntimeError(
                f"camera capture failed (libgphoto2: {exc}). Check autofocus "
                "(try manual focus or a lit, high-contrast subject), the memory "
                "card, and that the mode dial allows remote release (use "
                "P/Av/Tv/M, not movie/bulb)."
            )

        # Retry only the trigger itself after a reconnect: once the shutter has
        # actually fired, a blind retry would expose a second frame.
        cam = self._cam
        try:
            cam.trigger_capture()
        except gp.GPhoto2Error as exc:
            if not _device_gone(exc):
                raise _capture_error(exc) from exc
            LOGGER.warning(
                f"camera unreachable at capture ({exc}); reconnecting and "
                "retrying once"
            )
            self.reconnect()
            cam = self._cam
            try:
                cam.trigger_capture()
            except gp.GPhoto2Error as exc2:
                raise _capture_error(exc2) from exc2

        # Wait by real wall-clock time, not by counting event iterations:
        # wait_for_event returns *early* on each event, and bodies emit a burst
        # of non-file events (capture-complete, unknown) right after the shutter
        # trips. Crediting each of those the full step would blow the whole
        # settle budget in milliseconds and we'd give up before the card write
        # finishes. We cap each wait at the time remaining so we never overshoot.
        deadline = time.monotonic() + settle
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            event_type, event_data = cam.wait_for_event(min(500, int(remaining * 1000) + 1))
            if event_type == gp.GP_EVENT_FILE_ADDED:
                return event_data.folder.rstrip("/") + "/" + event_data.name

        # No path reported - the body wrote it to the card itself. Grab the
        # newest file there (same as `{"download": {"latest": true}}`).
        path = self.latest_image_file()
        if path is None:
            raise RuntimeError(
                "capture fired but no image appeared on the card - check "
                "autofocus (try manual focus or a lit, high-contrast subject), "
                "the memory card, and that the mode dial allows remote release."
            )
        return path

    @_retry_once_on_device_gone
    def delete(self, path: str) -> None:
        folder, name = os.path.split(path)
        self._cam.file_delete(folder, name)


class PTP(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug
    # option, or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "image-processing"), "ptp"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Create a new instance of this Camera component.

        ``EasyResource.new`` only constructs the instance - it does *not* call
        ``reconfigure``, and viam-server only calls ``reconfigure`` on later
        config changes, not on the initial add. So we must configure here, or
        ``self._session`` (and the other attributes set in ``reconfigure``)
        won't exist when the first DoCommand arrives.
        """
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate config. This model owns its USB device, so no dependencies."""
        attrs = struct_to_dict(config.attributes)

        download_dir = attrs.get("download_dir")
        if download_dir is not None and not isinstance(download_dir, str):
            raise ValueError("`download_dir` must be a string path")

        settle = attrs.get("capture_settle")
        if settle is not None and (
            not isinstance(settle, (int, float)) or isinstance(settle, bool) or settle < 0
        ):
            raise ValueError("`capture_settle` must be a non-negative number of seconds")

        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Read attributes and (re)open the USB camera connection."""
        attrs = struct_to_dict(config.attributes)

        self._port: Optional[str] = attrs.get("port") or None
        self._model_match: Optional[str] = attrs.get("camera_model") or None
        self._download_dir: Optional[str] = attrs.get("download_dir") or None
        self._delete_after_download: bool = bool(attrs.get("delete_after_download", False))
        # Seconds to wait after firing for the body to finish writing to the
        # card before we grab the "latest" file (bodies that don't report the
        # captured path - e.g. Canon writing to card). Bump it for slow cards
        # or flash recycle time.
        self._capture_settle: float = float(attrs.get("capture_settle", 2.0))

        if self._download_dir:
            os.makedirs(self._download_dir, exist_ok=True)

        # Serializes all camera access (libgphoto2 is single-owner) and tracks
        # which files we've already pulled this session for `new_only`.
        self._lock = asyncio.Lock()
        self._downloaded: set = set()

        # Drop any previous connection and lazily reopen on first use, so a
        # reconfigure after replugging the camera recovers cleanly.
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
        self._session = PTPSession(self._port, self._model_match)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run(self, fn, *args):
        """Run a blocking PTPSession call in a thread, holding the camera lock."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn, *args)

    def _save_to_dir(self, name: str, data: bytes) -> Optional[str]:
        """Write downloaded bytes to `download_dir` if configured; return path."""
        if not self._download_dir:
            return None
        dest = os.path.join(self._download_dir, name)
        with open(dest, "wb") as f:
            f.write(data)
        return dest

    # ------------------------------------------------------------------
    # Camera API
    # ------------------------------------------------------------------

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        """
        Return a single frame for the control tab: a live-view preview if the
        body supports it, otherwise the newest still on the card.
        """
        try:
            data = await self._run(self._session.preview)
            return [NamedImage("preview", data, CameraMimeType.JPEG)], ResponseMetadata()
        except Exception as exc:
            self.logger.debug(f"live-view preview unavailable ({exc}); using latest still")

        path = await self._run(self._session.latest_image_file)
        if not path:
            raise RuntimeError("camera has no live view and no images on storage")
        data = await self._run(self._session.read_file, path)
        name = os.path.basename(path)
        mime = _mime_for(name)
        return [NamedImage(name, data, mime)], ResponseMetadata()

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        return Camera.Properties(
            supports_pcd=False,
            intrinsic_parameters=None,
            distortion_parameters=None,
            mime_types=[CameraMimeType.JPEG],
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        resp: Dict[str, ValueTypes] = {}

        if "summary" in command:
            resp["summary"] = await self._summary()

        if "list_files" in command:
            resp["list_files"] = await self._list_files(command.get("list_files") or {})

        if "capture" in command:
            resp["capture"] = await self._capture(command.get("capture") or {})

        if "download" in command:
            resp["download"] = await self._download(command.get("download") or {})

        if "download_all" in command:
            resp["download_all"] = await self._download_all(command.get("download_all") or {})

        if "delete" in command:
            resp["delete"] = await self._delete(command.get("delete") or {})

        if not resp:
            raise ValueError(
                "no recognized command; supported: summary, list_files, "
                "capture, download, download_all, delete"
            )
        return resp

    # ------------------------------------------------------------------
    # DoCommand handlers
    # ------------------------------------------------------------------

    async def _summary(self) -> Mapping[str, ValueTypes]:
        text = await self._run(self._session.summary)
        return {
            "model": self._session.model_name,
            "port": self._session.port_path,
            "summary": text,
        }

    async def _list_files(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        files = await self._run(self._session.list_image_files)
        if opts.get("new_only"):
            files = [f for f in files if f not in self._downloaded]
        return {"files": files, "count": len(files)}

    async def _read_and_package(self, path: str) -> Dict[str, ValueTypes]:
        """Download `path`, optionally persist/delete, and build the response."""
        data = await self._run(self._session.read_file, path)
        name = os.path.basename(path)
        self._downloaded.add(path)

        saved_path = self._save_to_dir(name, data)
        if self._delete_after_download:
            await self._run(self._session.delete, path)

        return {
            "name": name,
            "path": path,
            "mime_type": _mime_for(name),
            "saved_to": saved_path,
            "size": len(data),
        }

    async def _capture(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """Trip the shutter, download the resulting still, return its metadata."""
        path = await self._run(self._session.capture, self._capture_settle)
        self.logger.info(f"captured {path}")
        return await self._read_and_package(path)

    async def _download(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """Download one file by `path`, or the newest with `latest: true`."""
        path = opts.get("path")
        if not path and opts.get("latest"):
            path = await self._run(self._session.latest_image_file)
        if not path:
            raise ValueError("`download` needs a `path`, or `latest: true`")
        return await self._read_and_package(str(path))

    async def _download_all(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Bulk-download to `download_dir` (required here - we don't base64 a whole
        card back over gRPC). Returns the saved paths. `new_only` skips files
        already pulled this session.
        """
        if not self._download_dir:
            raise ValueError(
                "`download_all` requires a `download_dir` to be configured"
            )

        files = await self._run(self._session.list_image_files)
        if opts.get("new_only"):
            files = [f for f in files if f not in self._downloaded]

        saved: List[str] = []
        for path in files:
            data = await self._run(self._session.read_file, path)
            dest = self._save_to_dir(os.path.basename(path), data)
            self._downloaded.add(path)
            if self._delete_after_download:
                await self._run(self._session.delete, path)
            if dest:
                saved.append(dest)

        self.logger.info(f"downloaded {len(saved)} file(s) to {self._download_dir}")
        return {"saved": saved, "count": len(saved)}

    async def _delete(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        path = opts.get("path")
        if not path:
            raise ValueError("`delete` needs a `path`")
        await self._run(self._session.delete, str(path))
        self._downloaded.discard(str(path))
        return {"deleted": str(path)}

    # ------------------------------------------------------------------
    # Unsupported camera methods
    # ------------------------------------------------------------------

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        raise NotImplementedError("PTP camera does not produce point clouds")

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return []

    async def close(self):
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
