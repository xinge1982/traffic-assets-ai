"""Freeze CLIP, extract labeled image features and train a linear classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .common import ClipEncoder, scan_dataset, split_indices


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_classifier(features, labels, regularization, max_iter, seed):
    if regularization <= 0 or max_iter < 1:
        raise ValueError("C 和 max-iter 必须大于 0")
    classifier = LogisticRegression(
        C=regularization, max_iter=max_iter, class_weight="balanced", random_state=seed,
    )
    return classifier.fit(features, labels)


def train(args):
    if args.C <= 0 or args.max_iter < 1 or args.batch_size < 1:
        raise ValueError("C、max-iter 和 batch-size 必须大于 0")
    classes, paths, labels = scan_dataset(args.data_dir)
    if args.val_dir:
        _, val_paths, val_labels = scan_dataset(args.val_dir, classes)
        train_idx = np.arange(len(paths))
        val_idx = np.arange(len(paths), len(paths) + len(val_paths))
        paths += val_paths
        labels = np.concatenate([labels, val_labels])
    else:
        train_idx, val_idx = split_indices(labels, args.val_ratio, args.seed)

    # Reject byte-identical images across splits and contradictory labels.
    hashes, seen = [], {}
    for path, label in zip(paths, labels):
        digest = fingerprint(path)
        if digest in seen and seen[digest] != int(label):
            raise ValueError(f"相同图片出现在不同类别中: {path}")
        seen[digest] = int(label)
        hashes.append(digest)
    train_hashes = {hashes[i] for i in train_idx}
    if any(hashes[i] in train_hashes for i in val_idx):
        raise ValueError("训练集和验证集有相同图片，请去重或使用独立 --val-dir")

    encoder = ClipEncoder(args.model, args.pretrained, args.device, args.batch_size)
    features = encoder.encode(paths)
    classifier = fit_classifier(features[train_idx], labels[train_idx], args.C, args.max_iter, args.seed)
    metrics = {
        "train_count": len(train_idx), "val_count": len(val_idx),
        "train_accuracy": float(classifier.score(features[train_idx], labels[train_idx])),
        "validation": None,
    }
    if len(val_idx):
        predicted = classifier.predict(features[val_idx])
        metrics["validation"] = {
            "accuracy": float(accuracy_score(labels[val_idx], predicted)),
            "report": classification_report(labels[val_idx], predicted, labels=list(range(len(classes))),
                                            target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels[val_idx], predicted,
                                                 labels=list(range(len(classes)))).tolist(),
        }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": 1, "model": args.model, "pretrained": args.pretrained,
        "preprocessing": "exif_alpha_white_square_pad_v1", "classes": classes,
        "feature_dim": int(features.shape[1]), "seed": args.seed,
        "C": args.C, "max_iter": args.max_iter,
    }
    joblib.dump({"config": config, "classifier": classifier}, output / "classifier.joblib")
    write_json(output / "classes.json", {name: i for i, name in enumerate(classes)})
    write_json(output / "config.json", config)
    write_json(output / "metrics.json", metrics)
    set_train = set(train_idx.tolist())
    write_json(output / "samples.json", [
        {"path": str(path.resolve()), "class_index": int(label), "class_name": classes[label],
         "sha256": digest, "split": "train" if i in set_train else "validation"}
        for i, (path, label, digest) in enumerate(zip(paths, labels, hashes))
    ])
    np.savez_compressed(output / "features.npz", features=features, labels=labels,
                        train_indices=train_idx, val_indices=val_idx)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"训练结果已保存: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="类别名称/图片 训练集根目录")
    parser.add_argument("--val-dir", type=Path, help="可选的独立验证集，类别目录须与训练集一致")
    parser.add_argument("--output-dir", type=Path, default=Path("sign_classification/output"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="OpenCLIP 权重名称或本地权重文件")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=10.0, help="分类头逆正则化系数")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
