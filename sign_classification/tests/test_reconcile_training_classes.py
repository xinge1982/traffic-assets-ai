import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sign_classification.reconcile_training_classes import move_no_overwrite, reconcile_training_classes


class ReconcileTrainingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train, self.review = self.root / "train", self.root / "review"
        self.train.mkdir()
        self.review.mkdir()

    def image(self, root, name, content=b"original training image"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def rows(self, result):
        with Path(result["report"]).open(encoding="utf-8-sig", newline="") as stream:
            return {row["pic"]: row for row in csv.DictReader(stream)}

    def test_moves_only_matching_misclassified_training_images(self):
        source = self.image(self.train, "旧类别/camera_a/图,一.png")
        reviewed = self.image(self.review, "新类别/图,一.png", b"different reviewed rendering")
        unchanged = self.image(self.train, "类别二/same.jpg")
        self.image(self.review, "类别二/same.jpg")
        untouched = self.image(self.train, "旧类别/only_train.png")
        self.image(self.review, "新类别/only_review.png")
        result = reconcile_training_classes(self.train, self.review)
        target = self.train / "新类别/camera_a/图,一.png"
        self.assertFalse(source.exists())
        self.assertEqual(target.read_bytes(), b"original training image")
        self.assertEqual(reviewed.read_bytes(), b"different reviewed rendering")
        self.assertTrue(unchanged.exists())
        self.assertTrue(untouched.exists())
        self.assertFalse((self.train / "新类别/only_review.png").exists())
        self.assertEqual(result["counts"], {"not_in_training": 1, "not_reviewed": 1, "unchanged": 1, "moved": 1})
        self.assertEqual(result["moves_by_class"], [{"old_class": "旧类别", "new_class": "新类别", "count": 1}])
        self.assertEqual(self.rows(result)["图,一.png"]["status"], "moved")
        self.assertTrue(Path(result["report"]).read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertTrue(Path(result["summary"]).exists())

    def test_dry_run_does_not_move_or_create_class_directory(self):
        original = self.image(self.train, "旧类别/a.png")
        self.image(self.review, "新类别/a.png")
        result = reconcile_training_classes(self.train, self.review, dry_run=True)
        self.assertTrue(original.exists())
        self.assertFalse((self.train / "新类别").exists())
        self.assertEqual(result["counts"], {"would_move": 1})
        self.assertEqual(result["moves_by_class"], [])

    def test_ambiguity_and_existing_destination_skipped(self):
        first = self.image(self.train, "类别一/duplicate.png")
        second = self.image(self.train, "类别二/duplicate.png")
        self.image(self.review, "类别三/duplicate.png")
        source = self.image(self.train, "类别一/nested/conflict.png")
        self.image(self.review, "类别三/conflict.png")
        target = self.train / "类别三/nested/conflict.png"
        target.mkdir(parents=True)
        result = reconcile_training_classes(self.train, self.review)
        self.assertEqual(result["counts"], {"conflict": 1, "ambiguous": 1})
        self.assertTrue(all(p.exists() for p in (first, second, source)))
        self.assertTrue(target.is_dir())

    def test_copy_failure_preserves_source_and_cleans_partial_target(self):
        source = self.image(self.train, "旧类别/a.png")
        self.image(self.review, "新类别/a.png")
        with patch("sign_classification.reconcile_training_classes.shutil.copyfileobj", side_effect=OSError("disk error")):
            result = reconcile_training_classes(self.train, self.review)
        self.assertEqual(result["counts"], {"error": 1})
        self.assertEqual(source.read_bytes(), b"original training image")
        self.assertFalse((self.train / "新类别/a.png").exists())
        self.assertEqual(self.rows(result)["a.png"]["message"], "disk error")

    def test_overlap_and_existing_report_fail_before_moves(self):
        source = self.image(self.train, "旧类别/a.png")
        self.image(self.review, "新类别/a.png")
        for review in (self.train, self.train / "nested", self.root):
            with self.assertRaises(ValueError):
                reconcile_training_classes(self.train, review)
        report = self.root / "existing.csv"
        report.write_text("keep old log")
        with self.assertRaises(FileExistsError):
            reconcile_training_classes(self.train, self.review, report)
        self.assertEqual(report.read_text(), "keep old log")
        self.assertTrue(source.exists())

    def test_interrupt_after_source_removed_keeps_destination(self):
        source = self.image(self.train, "旧类别/a.png")
        target = self.train / "新类别/a.png"
        target.parent.mkdir()
        unlink = Path.unlink

        def interrupt_after_unlink(path, *args, **kwargs):
            unlink(path, *args, **kwargs)
            if path == source:
                raise KeyboardInterrupt()

        with patch.object(Path, "unlink", interrupt_after_unlink):
            with self.assertRaises(KeyboardInterrupt):
                move_no_overwrite(source, target)
        self.assertFalse(source.exists())
        self.assertEqual(target.read_bytes(), b"original training image")


if __name__ == "__main__":
    unittest.main()
