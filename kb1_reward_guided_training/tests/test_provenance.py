"""Provenance guards for the KB1 two-stage workflow."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_provenance as provenance


class ProvenanceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root = root / 'kb1'
        self.root.mkdir()
        self.data = root / 'v2i_cleanned'
        for split in ('train', 'valid', 'test'):
            for kind in ('images', 'labels'):
                (self.data / split / kind).mkdir(parents=True)
            (self.data / split / 'images' / f'{split}.jpg').write_bytes(b'image-' + split.encode())
            (self.data / split / 'labels' / f'{split}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
        self.config = self.root / 'pest.yaml'
        self.config.write_text('path: ' + str(self.data).replace('\\', '/') + '\n')
        for key, value in (('ROOT', self.root), ('DATA_ROOT', self.data),
                           ('DATA_CONFIG', self.config)):
            patcher = patch.object(provenance, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_record_verify_and_detect_change(self):
        dataset = provenance.dataset_manifest()
        checkpoint = self.root / 'checkpoint_based' / 'yolov8n' / 'weights' / 'best.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint')
        provenance.record_supervised('yolov8n', 'yolov8n.pt', dataset,
                                     {'w3f': False, 'psa': False})
        provenance.verify_supervised('yolov8n', checkpoint)
        checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            provenance.verify_supervised('yolov8n', checkpoint)
        checkpoint.write_bytes(b'checkpoint')
        (self.data / 'train' / 'labels' / 'train.txt').write_text('0 0.4 0.5 0.2 0.2\n')
        with self.assertRaisesRegex(ValueError, 'Dataset changed'):
            provenance.verify_supervised('yolov8n', checkpoint)

    def test_reject_augmented_validation_and_unpaired_files(self):
        image = self.data / 'valid' / 'images' / 'valid_aug1_rot90cw.jpg'
        label = self.data / 'valid' / 'labels' / 'valid_aug1_rot90cw.txt'
        image.write_bytes(b'augmented')
        label.write_text('')
        with self.assertRaisesRegex(ValueError, 'augmented images'):
            provenance.dataset_manifest()
        image.unlink()
        label.unlink()
        (self.data / 'test' / 'labels' / 'test.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            provenance.dataset_manifest()


if __name__ == '__main__':
    unittest.main()
