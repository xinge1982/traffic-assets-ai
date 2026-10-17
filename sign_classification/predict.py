"""Classify a cropped sign image or a directory of sign images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .common import ClipEncoder, image_paths


def rank_predictions(classifier, features, classes, top_k, min_probability):
    probabilities = classifier.predict_proba(features)
    results = []
    for row in probabilities:
        order = np.argsort(-row)[:min(top_k, len(row))]
        ranked = [{"class_name": classes[int(classifier.classes_[i])],
                   "probability": float(row[i])} for i in order]
        results.append({"class_name": ranked[0]["class_name"],
                        "accepted": ranked[0]["probability"] >= min_probability,
                        "top_k": ranked})
    return results


def predict(args):
    if args.top_k < 1 or not 0 <= args.min_probability <= 1:
        raise ValueError("top-k 必须大于 0；min-probability 必须在 [0, 1] 内")
    # Only load artifacts produced by trusted training runs (joblib uses pickle).
    artifact = joblib.load(args.model_dir / "classifier.joblib")
    config = artifact["config"]
    if config["format_version"] != 1 or config["preprocessing"] != "exif_alpha_white_square_pad_v1":
        raise ValueError("不支持的模型版本或预处理配置")
    paths = image_paths(args.input)
    encoder = ClipEncoder(config["model"], args.pretrained or config["pretrained"],
                          args.device, args.batch_size)
    features = encoder.encode(paths)
    if features.shape[1] != config["feature_dim"]:
        raise ValueError("CLIP 特征维数与训练模型不一致")
    results = rank_predictions(artifact["classifier"], features, config["classes"],
                               args.top_k, args.min_probability)
    for path, result in zip(paths, results):
        result["path"] = str(path.resolve())
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("sign_classification/output"))
    parser.add_argument("--input", type=Path, required=True, help="单张图片或图片目录（递归）")
    parser.add_argument("--output", type=Path, help="输出 JSON 文件")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument("--pretrained", help="迁移机器时指定相同 CLIP 权重的本地路径")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
