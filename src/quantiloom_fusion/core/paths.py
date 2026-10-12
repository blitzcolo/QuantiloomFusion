"""Windows/WSL path translation: open-side native_path, record-side canonical_path."""

from __future__ import annotations

import re
from pathlib import Path

from .host import POSIX

_DRIVE = re.compile(r"^([A-Za-z]):/(.*)$")
_MOUNT = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def is_drive_path(s: str) -> bool:
    return bool(_DRIVE.match(s))


def native_path(value, posix=None):
    """Config or TOML path as written -> a Path this OS can open.

    On WSL a drive-letter path maps to /mnt/<drive>/; on Windows a /mnt path
    maps back to the drive. Relative paths pass through unchanged.
    """
    if posix is None:
        posix = POSIX
    s = str(value).replace("\\", "/")
    if posix:
        m = _DRIVE.match(s)
        if m:
            s = "/mnt/" + m.group(1).lower() + "/" + m.group(2)
    else:
        m = _MOUNT.match(s)
        if m:
            s = m.group(1).upper() + ":/" + m.group(2)
    return Path(s)


def canonical_path(p, posix=None):
    """Form recorded in plans and embedded in job TOMLs.

    A file under a /mnt/<drive>/ mount is recorded with its Windows spelling,
    so recipe fingerprints and job hashes are identical on Windows and WSL.
    Other POSIX paths are recorded as-is (they cannot reach the Windows CLI).
    """
    if posix is None:
        posix = POSIX
    s = Path(p).as_posix()
    if posix:
        m = _MOUNT.match(s)
        if m:
            return m.group(1).upper() + ":/" + m.group(2)
    return s
