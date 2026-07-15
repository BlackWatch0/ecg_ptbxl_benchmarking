import csv
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import prepare_original_models_benchmark_data as prepare


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_record(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".hea").write_text("synthetic header\n")
    path.with_suffix(".dat").write_bytes(b"synthetic data")


class PrepareOriginalModelsBenchmarkDataTest(unittest.TestCase):
    def test_snr_name_normalization(self):
        self.assertEqual(prepare._snr_from_text("mixed_snr24"), 24)
        self.assertEqual(prepare._snr_from_text("mixed_snrm6"), -6)
        self.assertEqual(prepare._snr_from_text("mixed_snr_-6"), -6)

    def test_real_archive_manifest_path_columns(self):
        noisy = prepare._csv_layout_from_fields([
            "ecg_id", "filename_lr", "signal_type", "snr_target_db",
            "wfdb_record_relative"
        ])
        denoised = prepare._csv_layout_from_fields([
            "ecg_id", "snr_target_db", "final_wfdb_record_relative",
            "wfdb_export_ok"
        ])
        self.assertEqual(noisy[2], "wfdb_record_relative")
        self.assertEqual(denoised[2], "final_wfdb_record_relative")

    def test_safe_extract_is_reused_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("data/file.txt", "ok")
            first = prepare.inspect_and_extract(archive, root / "workspace")
            second = prepare.inspect_and_extract(archive, root / "workspace")
            self.assertEqual(first, second)
            self.assertEqual((first / "data/file.txt").read_text(), "ok")

            unsafe = root / "unsafe.tar"
            with tarfile.open(unsafe, "w") as output:
                member = tarfile.TarInfo("../escape.txt")
                member.size = 3
                output.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                prepare.inspect_and_extract(unsafe, root / "unsafe_workspace")
            self.assertFalse((root / "escape.txt").exists())

    def test_discovers_and_validates_fold10_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records"
            clean_rows = []
            for record_id in (1, 2):
                relative = "clean/{:05d}_lr".format(record_id)
                write_record(root, relative)
                clean_rows.append({"ecg_id": record_id, "strat_fold": 10, "filename_lr": relative})
            write_csv(root / "ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv",
                      ["ecg_id", "strat_fold", "filename_lr"], clean_rows)

            for condition in ("noisy", "denoised"):
                rows = []
                for snr in prepare.SNRS:
                    for record_id in (1, 2):
                        relative = "{}/snr{}/{:05d}".format(condition, "m6" if snr == -6 else snr, record_id)
                        write_record(root, relative)
                        rows.append({"ecg_id": record_id, "snr_target_db": snr,
                                     "condition": condition, "record_path": relative})
                write_csv(root / condition / "manifest.csv",
                          ["ecg_id", "snr_target_db", "condition", "record_path"], rows)

            config_path = prepare.discover_and_validate([root], records)
            config = json.loads(config_path.read_text())
            self.assertEqual(config["fold10_record_count"], 2)
            self.assertEqual(config["snrs_db"], [24, 12, 6, 0, -6])
            for condition, expected_rows in (("clean", 2), ("noisy", 10), ("denoised", 10)):
                with Path(config["manifests"][condition]).open() as source:
                    self.assertEqual(sum(1 for _ in csv.DictReader(source)), expected_rows)

    def test_missing_denoised_snr_fails_coverage_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_record(root, "clean/00001_lr")
            write_csv(root / "ptbxl_database.csv", ["ecg_id", "strat_fold", "filename_lr"],
                      [{"ecg_id": 1, "strat_fold": 10, "filename_lr": "clean/00001_lr"}])
            for condition in ("noisy", "denoised"):
                rows = []
                snrs = prepare.SNRS if condition == "noisy" else prepare.SNRS[:-1]
                for snr in snrs:
                    relative = "{}/snr{}/00001".format(condition, snr)
                    write_record(root, relative)
                    rows.append({"ecg_id": 1, "snr_db": snr, "condition": condition,
                                 "record_path": relative})
                write_csv(root / condition / "manifest.csv",
                          ["ecg_id", "snr_db", "condition", "record_path"], rows)
            with self.assertRaisesRegex(ValueError, "denoised SNR -6.*1 missing"):
                prepare.discover_and_validate([root], root / "output")


if __name__ == "__main__":
    unittest.main()
