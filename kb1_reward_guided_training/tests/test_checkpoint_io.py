"""Windows-origin trainer paths must load without changing global path behavior."""
import io
import os
import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from checkpoint_io import windows_checkpoint_paths


class WindowsOriginPath:
    def __reduce__(self):
        return pathlib.WindowsPath, ('D:/old/project/best.pt',)


@unittest.skipIf(os.name == 'nt', 'POSIX relocation contract')
class CheckpointPathTests(unittest.TestCase):
    def test_unpickle_windows_metadata_and_restore_class(self):
        original = pathlib.WindowsPath
        buffer = io.BytesIO()
        torch.save({'path': WindowsOriginPath()}, buffer)
        buffer.seek(0)
        with windows_checkpoint_paths():
            loaded = torch.load(buffer, weights_only=False)
        self.assertEqual(str(loaded['path']), 'D:\\old\\project\\best.pt')
        self.assertIs(pathlib.WindowsPath, original)

    def test_restore_after_nested_exception(self):
        original = pathlib.WindowsPath
        with self.assertRaises(RuntimeError):
            with windows_checkpoint_paths():
                with windows_checkpoint_paths():
                    raise RuntimeError('loader failed')
        self.assertIs(pathlib.WindowsPath, original)
