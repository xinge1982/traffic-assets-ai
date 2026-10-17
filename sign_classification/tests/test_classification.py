import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from sign_classification.common import load_image, scan_dataset, split_indices
from sign_classification.predict import predict, rank_predictions
from sign_classification.train import fit_classifier, train


class FakeEncoder:
    """Fixed features to test pipeline without downloading CLIP."""
    def __init__(self, *args):
        pass

    def encode(self, paths):
        return np.asarray([[1., 0.] if "禁止左转" in p.parts else [0., 1.]
                           for p in paths], dtype=np.float32)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for label, name in enumerate(["禁止左转", "禁止右转"]):
            folder = self.root / "train" / name
            folder.mkdir(parents=True)
            for i in range(5):
                Image.new("RGB", (20 + i, 10), (label * 100, i * 20, 0)).save(folder / f"{i}.png")

    def args(self, **kwargs):
        values = dict(data_dir=self.root / "train", val_dir=None, val_ratio=.2,
                      seed=42, C=10., max_iter=1000, batch_size=2, model="ViT-B-32",
                      pretrained="test", device="cpu", output_dir=self.root / "model")
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_scan_split(self):
        classes, paths, labels = scan_dataset(self.root / "train")
        self.assertEqual(classes, sorted(["禁止左转", "禁止右转"]))
        self.assertEqual(len(paths), 10)
        a, b = split_indices(labels, .2, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(10)))
        self.assertEqual(set(labels[b]), {0, 1})
        self.assertTrue(np.array_equal(b, split_indices(labels, .2, 42)[1]))
        with self.assertRaises(ValueError):
            split_indices(np.asarray([0, 1]), .2, 42)
        self.assertEqual(len(split_indices(np.asarray([0, 1]), 0, 42)[1]), 0)

    def test_padding_alpha(self):
        path = self.root / "alpha.png"
        Image.new("RGBA", (40, 10), (255, 0, 0, 0)).save(path)
        image = load_image(path)
        self.assertEqual(image.size, (40, 40))
        self.assertEqual(image.getpixel((20, 20)), (255, 255, 255))
        Image.new("RGB", (40, 10), "red").save(path)
        image = load_image(path)
        self.assertEqual(image.getpixel((0, 15)), (255, 0, 0))
        self.assertEqual(image.getpixel((39, 24)), (255, 0, 0))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    @patch("sign_classification.train.ClipEncoder", FakeEncoder)
    def test_pipeline(self):
        args = self.args()
        train(args)
        artifact = joblib.load(args.output_dir / "classifier.joblib")
        metrics = json.loads((args.output_dir / "metrics.json").read_text())
        self.assertEqual(metrics["validation"]["accuracy"], 1.)
        with np.load(args.output_dir / "features.npz") as features:
            self.assertEqual(features["features"].shape, (10, 2))
        ranked = rank_predictions(artifact["classifier"], np.asarray([[1., 0.]]),
                                  artifact["config"]["classes"], 50, 1.)
        self.assertEqual(len(ranked[0]["top_k"]), 2)
        self.assertEqual(ranked[0]["class_name"], "禁止左转")
        self.assertFalse(ranked[0]["accepted"])
        with patch("sign_classification.predict.ClipEncoder", FakeEncoder):
            predict(argparse.Namespace(model_dir=args.output_dir, input=args.data_dir,
                    pretrained=None, device="cpu", batch_size=2, top_k=1,
                    min_probability=0., output=self.root / "predictions.json"))
        results = json.loads((self.root / "predictions.json").read_text())
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r["class_name"] in Path(r["path"]).parts for r in results))

    @patch("sign_classification.train.ClipEncoder", FakeEncoder)
    def test_no_validation_and_duplicates(self):
        args = self.args(val_ratio=0)
        train(args)
        self.assertIsNone(json.loads((args.output_dir / "metrics.json").read_text())["validation"])
        val = self.root / "val"
        for folder in args.data_dir.iterdir():
            target = val / folder.name
            target.mkdir(parents=True)
            (target / "duplicate.png").write_bytes((folder / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "相同图片"):
            train(self.args(val_dir=val))

    def test_invalid_dataset(self):
        (self.root / "train" / "空目录").mkdir()
        with self.assertRaisesRegex(ValueError, "没有图片"):
            scan_dataset(self.root / "train")
        with self.assertRaisesRegex(ValueError, "验证集类别"):
            scan_dataset(self.root / "train", ["other"])

    def test_multiclass_head(self):
        features = np.tile(np.eye(3, dtype=np.float32), (4, 1))
        labels = np.tile(np.arange(3), 4)
        head = fit_classifier(features, labels, 10., 1000, 42)
        ranked = rank_predictions(head, np.eye(3), ["左转", "右转", "限速60"], 2, 0.)
        self.assertEqual([r["class_name"] for r in ranked], ["左转", "右转", "限速60"])
        self.assertTrue(all(len(r["top_k"]) == 2 for r in ranked))


if __name__ == "__main__":
    unittest.main()
