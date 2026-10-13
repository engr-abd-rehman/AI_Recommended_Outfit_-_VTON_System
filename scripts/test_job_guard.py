"""Check single-job admission without loading diffusion models."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app as server


class JobGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.upload = root / "upload.png"
        self.catalog = root / "catalog.png"
        self.upload.write_bytes(b"route fixture")
        self.catalog.write_bytes(b"route fixture")
        replacements = patch.multiple(server, CATALOG_DIR=str(root), UPLOAD_DIR=str(root), _jobs={})
        replacements.start()
        self.addCleanup(replacements.stop)
        worker = patch.object(server.threading, "Thread")
        self.worker = worker.start()
        self.addCleanup(worker.stop)
        self.client = server.app.test_client()
        self.payload = {"uploaded_filepath": str(self.upload), "catalog_filename": self.catalog.name,
                        "uploaded_type": "upper"}

    def test_second_running_job_is_rejected(self):
        first = self.client.post("/tryon", json=self.payload)
        second = self.client.post("/tryon", json=self.payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(self.worker.call_count, 1)

    def test_completed_job_releases_admission(self):
        first = self.client.post("/tryon", json=self.payload).get_json()
        server._update_job(first["job_id"], status="done")
        self.assertEqual(self.client.post("/tryon", json=self.payload).status_code, 200)

    def test_worker_start_failure_does_not_block_future_jobs(self):
        self.worker.return_value.start.side_effect = RuntimeError("cannot start thread")
        self.assertEqual(self.client.post("/tryon", json=self.payload).status_code, 500)
        self.worker.return_value.start.side_effect = None
        self.assertEqual(self.client.post("/tryon", json=self.payload).status_code, 200)


if __name__ == "__main__":
    unittest.main()
