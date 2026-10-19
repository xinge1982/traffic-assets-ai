import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from sign_classification.classify_directory import classify_directory, validate_classes
from sign_classification.common import load_image
from sign_classification.train import fit_classifier


class PixelEncoder:
    """Real image loading, deterministic features instead of downloading CLIP."""
    def __init__(self, *args):
        self.calls = []

    def encode(self, paths):
        self.calls.append(len(paths))
        rows = []
        for path in paths:
            with load_image(path) as image:
                rgb = image.getpixel((image.width // 2, image.height // 2))
                rows.append([1., 0.] if rgb[0] > rgb[2] else [0., 1.])
        return np.asarray(rows, dtype=np.float32)


class DirectoryClassificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "input"
        for folder, color in [("camera_a", "red"), ("camera_b", "red"), ("camera_c", "blue")]:
            target = self.source / folder / "same.png"
            target.parent.mkdir(parents=True)
            Image.new("RGB", (20, 10), color).save(target)
        (self.source / "notes.txt").write_text("ignore non-images")
        self.model = self.root / "model"
        self.model.mkdir()
        features = np.tile(np.eye(2, dtype=np.float32), (5, 1))
        labels = np.tile(np.arange(2), 5)
        self.artifact = {
            "classifier": fit_classifier(features, labels, 10., 1000, 42),
            "config": {"format_version": 1, "preprocessing": "exif_alpha_white_square_pad_v1",
                       "model": "ViT-B-32", "pretrained": "training-weight", "feature_dim": 2,
                       "classes": ["10100_停车让行标志", "10200_减速让行标志"]},
        }
        self.save_model()

    def save_model(self):
        joblib.dump(self.artifact, self.model / "classifier.joblib")

    def args(self, **changes):
        values = dict(input=self.source, output=self.root / "sorted", model_dir=self.model,
                      pretrained="local-weights.bin", device="cpu", batch_size=2, top_k=3,
                      min_probability=0., skip_errors=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def run_sort(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return classify_directory(args)

    def test_full_pipeline_preserves_files_and_batches(self):
        encoder = PixelEncoder()
        args = self.args()
        args.output.mkdir()  # An existing empty output is permitted.
        with patch("sign_classification.classify_directory.ClipEncoder", return_value=encoder) as factory:
            summary = self.run_sort(args)
        factory.assert_called_once_with("ViT-B-32", "local-weights.bin", "cpu", 2)
        self.assertEqual(encoder.calls, [2, 1])
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["classified"], 3)
        rows = [json.loads(line) for line in Path(summary["report"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            original, copied = Path(row["path"]), Path(row["output_path"])
            expected = self.artifact["config"]["classes"][1 if "camera_c" in original.parts else 0]
            self.assertEqual(row["class_name"], expected)
            self.assertEqual(copied.relative_to(args.output).parts[0], expected)
            self.assertEqual(copied.name, original.name)
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertEqual(len(row["top_k"]), 2)
        self.assertEqual(len({row["output_path"] for row in rows}), 3)
        csv_path = Path(summary["csv"])
        self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            csv_rows = list(reader)
            self.assertEqual(reader.fieldnames, ["pic", "class", "confidence"])
        self.assertEqual(len(csv_rows), len(rows))
        for actual, expected in zip(csv_rows, rows):
            self.assertEqual(actual["pic"], Path(expected["path"]).name)
            self.assertEqual(actual["class"], expected["class_name"])
            self.assertEqual(float(actual["confidence"]), expected["probability"])
        self.assertEqual(json.loads((args.output / "summary.json").read_text(encoding="utf-8")), summary)

    @patch("sign_classification.classify_directory.ClipEncoder", PixelEncoder)
    def test_threshold_routes_to_review(self):
        args = self.args(min_probability=1.)
        summary = self.run_sort(args)
        self.assertEqual(summary["review"], 3)
        self.assertEqual(summary["classified"], 0)
        self.assertEqual(len(list((args.output / "_待复核").rglob("*.png"))), 3)
        with Path(summary["csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row["class"] in self.artifact["config"]["classes"] for row in rows))
        self.assertTrue(all(float(row["confidence"]) < 1. for row in rows))

    @patch("sign_classification.classify_directory.ClipEncoder", PixelEncoder)
    def test_bad_image_skipped_without_discarding_good_batch_members(self):
        (self.source / "camera_a" / "bad.png").write_bytes(b"not an image")
        args = self.args(skip_errors=True, batch_size=3)
        summary = self.run_sort(args)
        self.assertEqual(summary["classified"], 3)
        self.assertEqual(summary["failed"], 1)
        rows = [json.loads(line) for line in Path(summary["report"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["status"] == "error" for row in rows), 1)
        with Path(summary["csv"]).open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 4)
        self.assertEqual([r for r in csv_rows if r["pic"] == "bad.png"],
                         [{"pic": "bad.png", "class": "", "confidence": ""}])

    @patch("sign_classification.classify_directory.ClipEncoder", PixelEncoder)
    def test_csv_quotes_chinese_filename_with_comma(self):
        filename = '标志,一号.png'
        Image.new("RGB", (20, 10), "red").save(self.source / filename)
        summary = self.run_sort(self.args())
        with Path(summary["csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        row = next(row for row in rows if row["pic"] == filename)
        self.assertEqual(row["class"], self.artifact["config"]["classes"][0])
        self.assertGreater(float(row["confidence"]), 0.)

    def test_invalid_paths_do_not_modify_input_or_output(self):
        for output in [self.source, self.source / "nested"]:
            with self.assertRaises(ValueError):
                self.run_sort(self.args(output=output))
        output = self.root / "nonempty"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            self.run_sort(self.args(output=output))
        self.assertEqual(sentinel.read_text(), "keep")

    def test_unsafe_class_directory_rejected_before_encoder_load(self):
        for name in ["../escape", "a/b", "C:folder", "_待复核"]:
            with self.assertRaises(ValueError):
                validate_classes(["valid", name])
        self.artifact["config"]["classes"] = ["valid", "../escape"]
        self.save_model()
        with patch("sign_classification.classify_directory.ClipEncoder") as factory:
            with self.assertRaises(ValueError):
                self.run_sort(self.args())
        factory.assert_not_called()
        self.assertFalse(self.args().output.exists())

    @patch("sign_classification.classify_directory.ClipEncoder", PixelEncoder)
    def test_feature_mismatch_stops(self):
        self.artifact["config"]["feature_dim"] = 3
        self.save_model()
        with self.assertRaisesRegex(ValueError, "特征维数"):
            self.run_sort(self.args())


if __name__ == "__main__":
    unittest.main()
