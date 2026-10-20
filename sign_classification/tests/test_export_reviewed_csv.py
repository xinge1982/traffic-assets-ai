import csv
from pathlib import Path
import tempfile
import unittest

from sign_classification.export_reviewed_csv import export_reviewed_csv


class ReviewedCSVTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def add_file(self, name, content=b"image bytes unchanged"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def read_rows(self):
        with (self.root / "classification_results.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["pic", "class"])
            return list(reader)

    def test_exports_nested_images_with_first_level_class(self):
        image = self.add_file("10100_停车让行标志/camera_a/图,一.PNG")
        self.add_file("10200_减速让行标志/002.jpg")
        self.add_file("10100_停车让行标志/notes.txt")
        self.add_file("root_image.png")  # No class: root-level images are ignored.
        result = export_reviewed_csv(self.root)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["classes"], 2)
        self.assertEqual(self.read_rows(), [
            {"pic": "图,一.PNG", "class": "10100_停车让行标志"},
            {"pic": "002.jpg", "class": "10200_减速让行标志"},
        ])
        self.assertTrue(Path(result["output"]).read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(image.read_bytes(), b"image bytes unchanged")

    def test_regenerates_existing_csv_after_manual_reclassification(self):
        original = self.add_file("类别一/001.png")
        output = self.add_file("classification_results.csv", b"pic,class,confidence\nold.png,old,0.9\n")
        export_reviewed_csv(self.root)
        moved = self.root / "类别二" / "001.png"
        moved.parent.mkdir()
        original.rename(moved)
        export_reviewed_csv(self.root)
        self.assertEqual(self.read_rows(), [{"pic": "001.png", "class": "类别二"}])
        self.assertEqual(output.name, "classification_results.csv")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_duplicate_names_retained_and_empty_input_preserves_csv(self):
        self.add_file("类别一/same.png")
        self.add_file("类别二/same.png")
        result = export_reviewed_csv(self.root)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["duplicate_names"], 1)
        self.assertEqual(len(self.read_rows()), 2)
        for path in self.root.rglob("*.png"):
            path.unlink()
        output = self.root / "classification_results.csv"
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "没有找到图片"):
            export_reviewed_csv(self.root)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
