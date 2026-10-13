"""Offline safety checks for the asset restoration utility."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import restore_assets


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / ".downloads"
        self.cache.mkdir()
        (self.root / "catalog_metadata.json").write_text(json.dumps([{"filename": "catalog.png"}]))
        self.patch = patch.multiple(restore_assets, ROOT=self.root, CACHE=self.cache)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def archive(self, entries):
        with zipfile.ZipFile(self.cache / "text2human-dataset.zip", "w") as file:
            for name, data in entries.items():
                file.writestr(name, data)

    def test_selective_extraction_preserves_catalog(self):
        self.archive({"data/train_images/catalog.png": b"image",
                      "data/train_images/unindexed.png": b"skip",
                      "data/segm/catalog_segm.png": b"mask",
                      "data/shape_ann/train_ann_file.txt": b"annotations",
                      "data/densepose/unused.png": b"skip"})
        restore_assets.restore_dataset(extract_only=True)
        self.assertEqual((self.root / "Kaggle_datasets/train_images/catalog.png").read_bytes(), b"image")
        self.assertFalse((self.root / "Kaggle_datasets/train_images/unindexed.png").exists())
        self.assertFalse((self.root / "Kaggle_datasets/densepose").exists())
        report = json.loads((self.cache / "dataset-manifest.json").read_text())
        self.assertEqual(report["catalog_images"], 1)
        self.assertEqual(report["missing_masks"], [])

    def test_wrong_dataset_rejected_before_extraction(self):
        self.archive({"data/train_images/wrong.png": b"image"})
        with self.assertRaisesRegex(ValueError, "does not match catalog"):
            restore_assets.restore_dataset(extract_only=True)
        self.assertFalse((self.root / "Kaggle_datasets").exists())

    def test_fresh_clone_without_prebuilt_metadata(self):
        (self.root / "catalog_metadata.json").unlink()
        self.archive({"data/train_images/catalog.png": b"image",
                      "data/train_images/second.png": b"image2",
                      "data/segm/catalog_segm.png": b"mask",
                      "data/segm/second_segm.png": b"mask2",
                      "data/segm/test_only_segm.png": b"unused"})
        restore_assets.restore_dataset(extract_only=True)
        report = json.loads((self.cache / "dataset-manifest.json").read_text())
        self.assertEqual(report["catalog_images"], 2)
        self.assertFalse((self.root / "Kaggle_datasets/segm/test_only_segm.png").exists())

    def test_missing_mask_rejected_before_extraction(self):
        self.archive({"data/train_images/catalog.png": b"image"})
        with self.assertRaisesRegex(ValueError, "segmentation masks"):
            restore_assets.restore_dataset(extract_only=True)
        self.assertFalse((self.root / "Kaggle_datasets").exists())

    def test_archive_traversal_rejected(self):
        self.archive({"data/train_images/../../outside.png": b"unsafe"})
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            restore_assets.restore_dataset(extract_only=True)

    def test_hashes(self):
        path = self.root / "blob"
        path.write_bytes(b"test")
        self.assertEqual(restore_assets.digest(path), hashlib.sha256(b"test").hexdigest())
        self.assertEqual(restore_assets.digest(path, "sha1", git_blob=True), hashlib.sha1(b"blob 4\0test").hexdigest())

    def test_interrupted_download_resumes_at_exact_offset(self):
        class Response:
            def __init__(self, status, chunks, headers=None):
                self.status_code, self.chunks, self.headers = status, chunks, headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, size):
                for chunk in self.chunks:
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk

        entry = SimpleNamespace(rfilename="checkpoints/ootd/config.json", size=4,
                                lfs={"sha256": hashlib.sha256(b"test").hexdigest()}, blob_id=None)
        responses = [Response(200, [b"te", restore_assets.requests.exceptions.ChunkedEncodingError()]),
                     Response(206, [b"st"], {"Content-Range": "bytes 2-3/4"})]
        checkpoints = self.root / "OOTDiffusion/checkpoints"
        with patch.object(restore_assets, "CHECKPOINTS", checkpoints), \
             patch.object(restore_assets, "HfApi") as api, \
             patch.object(restore_assets.requests, "get", side_effect=responses) as request, \
             patch.object(restore_assets.time, "sleep"):
            api.return_value.model_info.side_effect = [SimpleNamespace(siblings=[entry]), SimpleNamespace(siblings=[])]
            restore_assets.download_models(workers=1)
        self.assertEqual(request.call_args_list[1].kwargs["headers"], {"Range": "bytes=2-"})
        self.assertEqual((checkpoints / "ootd/config.json").read_bytes(), b"test")
        self.assertFalse((checkpoints / "ootd/config.json.part").exists())


if __name__ == "__main__":
    unittest.main()
