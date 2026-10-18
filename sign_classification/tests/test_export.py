import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from sign_classification.export_onnx import head_parameters


class ExportHeadTests(unittest.TestCase):
    def test_binary_and_multiclass_probabilities(self):
        rng = np.random.default_rng(42)
        for count in (2, 3):
            x = rng.normal(size=(60, 8))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y = np.arange(60) % count
            classifier = LogisticRegression().fit(x, y)
            names = [f"类别{i}" for i in range(count)]
            weight, bias, ordered = head_parameters(classifier, names, 8)
            logits = x @ weight.T + bias
            if count == 2:
                positive = 1 / (1 + np.exp(-logits))
                actual = np.concatenate([1 - positive, positive], axis=1)
            else:
                actual = np.exp(logits - logits.max(axis=1, keepdims=True))
                actual /= actual.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(actual, classifier.predict_proba(x), atol=1e-7)
            self.assertEqual(names, ordered)

    def test_reject_mismatched_classes(self):
        classifier = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
        with self.assertRaises(ValueError):
            head_parameters(classifier, ["a", "b", "c"], 2)


if __name__ == "__main__":
    unittest.main()
