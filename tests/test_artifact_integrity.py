"""Historical evidence receipts must survive Git and cross-platform clones."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def verify_receipt(manifest):
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for name, expected in entries.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Receipt must reference sibling files only")
        actual = hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact bytes changed: {manifest.parent.name}/{name}")
    return len(entries)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_all_published_historical_receipts(self):
        manifests = list((ROOT / "artifacts").glob("*/SHA256SUMS.json"))
        self.assertGreaterEqual(len(manifests), 10)
        self.assertGreaterEqual(sum(verify_receipt(m) for m in manifests), 51)

    def test_line_ending_change_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = b"evidence\r\n"
            (root / "result.txt").write_bytes(original)
            manifest = root / "SHA256SUMS.json"
            manifest.write_text(json.dumps({"result.txt": hashlib.sha256(original).hexdigest()}))
            self.assertEqual(verify_receipt(manifest), 1)
            (root / "result.txt").write_bytes(b"evidence\n")
            with self.assertRaises(ValueError):
                verify_receipt(manifest)


if __name__ == "__main__":
    unittest.main()
