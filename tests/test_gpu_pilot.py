"""Offline pilot tests. These do not import torch or execute a model."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gpu_pilot.py"
spec = importlib.util.spec_from_file_location("gpu_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_eight_distinct_synthetic_cases(self):
        self.assertEqual(len(pilot.cases()), 8)
        self.assertEqual(len({pilot.digest(c) for c in pilot.cases()}), 8)

    def test_output_is_parsed_not_assumed_valid(self):
        case = pilot.cases()[0]
        self.assertEqual(pilot.validate_output('{"step":0,"amount":2,"version":0}', case), (True, None))
        for invalid in ['prose', '{"step":0,"amount":2,"version":true}', '{"step":0,"amount":999,"version":0}']:
            self.assertFalse(pilot.validate_output(invalid, case)[0])

    def test_existing_evidence_directory_is_never_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "run"
            pilot.new_run_directory(target)
            with self.assertRaises(FileExistsError):
                pilot.new_run_directory(target)

    def test_receipt_saved_without_leftover_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            pilot.save_receipt(target, {"status": "interrupted_or_failed"})
            self.assertIn("interrupted_or_failed", target.read_text())
            self.assertFalse(target.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
