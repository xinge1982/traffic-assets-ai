import csv
from pathlib import Path
import tempfile
import unittest

from sign_classification.compare_classification_dirs import compare_directories


class CompareDirectoriesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old, self.new = self.root / "old", self.root / "reviewed"
        self.old.mkdir()
        self.new.mkdir()

    def image(self, root, path):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"unchanged image bytes")
        return target

    def read_rows(self, result):
        with Path(result["output"]).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_class_changes_additions_and_removals(self):
        old = self.image(self.old, "旧类别/标志,一.PNG")
        self.image(self.new, "新类别/标志,一.PNG")
        self.image(self.old, "类别二/camera_a/unchanged.png")
        self.image(self.new, "类别二/camera_b/unchanged.png")
        self.image(self.old, "类别二/removed.jpg")
        self.image(self.new, "类别二/added.jpg")
        self.image(self.new, "root-image.png")  # No category, ignored.
        result = compare_directories(self.old, self.new)
        self.assertEqual((result["old_images"], result["new_images"]), (3, 3))
        rows = {r["pic"]: r for r in self.read_rows(result)}
        self.assertEqual(result["exported_rows"], 3)
        self.assertEqual(rows["标志,一.PNG"]["status"], "class_changed")
        self.assertEqual(rows["标志,一.PNG"]["old_class"], "旧类别")
        self.assertEqual(rows["标志,一.PNG"]["new_class"], "新类别")
        self.assertEqual(rows["added.jpg"]["status"], "only_new")
        self.assertEqual(rows["added.jpg"]["old_path"], "")
        self.assertEqual(rows["removed.jpg"]["status"], "only_old")
        self.assertNotIn("unchanged.png", rows)
        self.assertEqual(old.read_bytes(), b"unchanged image bytes")
        self.assertTrue(Path(result["output"]).read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_duplicates_are_not_guessed(self):
        self.image(self.old, "类别一/camera_a/same.png")
        self.image(self.old, "类别二/camera_b/same.png")
        self.image(self.new, "类别三/same.png")
        result = compare_directories(self.old, self.new)
        rows = self.read_rows(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ambiguous")
        self.assertEqual(rows[0]["old_class"], "类别一 | 类别二")
        self.assertIn("camera_a/same.png", rows[0]["old_path"])
        self.assertIn("camera_b/same.png", rows[0]["old_path"])
        self.assertNotIn("class_changed", result["counts"])

    def test_no_differences_and_include_unchanged(self):
        self.image(self.old, "类别一/001.png")
        self.image(self.new, "类别一/001.png")
        output = self.root / "reports" / "diff.csv"
        result = compare_directories(self.old, self.new, output)
        self.assertEqual(self.read_rows(result), [])
        result = compare_directories(self.old, self.new, output, include_unchanged=True)
        self.assertEqual(self.read_rows(result)[0]["status"], "unchanged")
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            compare_directories(self.root / "missing", self.new, output)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
