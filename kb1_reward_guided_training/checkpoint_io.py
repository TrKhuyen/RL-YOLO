"""Scoped loading of trusted Windows-origin checkpoints on POSIX systems."""
from contextlib import contextmanager
import os
import pathlib
import threading

_PATH_LOCK = threading.RLock()


@contextmanager
def windows_checkpoint_paths():
    """Preserve stored Windows path strings as PureWindowsPath when unpickling.

    Historical YOLOv5 checkpoints contain WindowsPath objects in trainer metadata.
    No file is converted or rewritten. The patch ends before model execution.
    """
    if os.name == 'nt':
        yield
        return
    with _PATH_LOCK:
        original = pathlib.WindowsPath
        try:
            pathlib.WindowsPath = pathlib.PureWindowsPath
            yield
        finally:
            pathlib.WindowsPath = original
