"""递归分类图片，并复制到输出目录中对应的类别文件夹。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

if __package__:
    from .common import ClipEncoder, image_paths
    from .predict import rank_predictions
else:
    # Also support PyCharm's Script path mode and direct `python .../file.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sign_classification.common import ClipEncoder, image_paths
    from sign_classification.predict import rank_predictions


REVIEW_FOLDER = "_待复核"


def validate_classes(classes):
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("训练配置至少需要两个类别")
    seen = set()
    for name in classes:
        if (not isinstance(name, str) or not name or name in (".", "..")
                or name.endswith((" ", "."))
                or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name)):
            raise ValueError(f"类别名称不能作为目录名：{name!r}")
        key = name.casefold()
        if key in seen or key == REVIEW_FOLDER.casefold():
            raise ValueError(f"类别重名或使用了保留名称：{name}")
        seen.add(key)


def predict_batch(encoder, classifier, config, paths, args):
    features = encoder.encode(paths)
    if features.shape != (len(paths), config["feature_dim"]) or not np.isfinite(features).all():
        raise ValueError("CLIP 特征维数或数值与训练模型不一致")
    results = rank_predictions(classifier, features, config["classes"],
                               args.top_k, args.min_probability)
    if len(results) != len(paths):
        raise ValueError("预测结果数量与图片数量不一致")
    for result in results:
        probabilities = [r["probability"] for r in result["top_k"]]
        if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("分类头输出了无效概率")
    return results


def classify_directory(args):
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size 和 top-k 必须大于 0")
    if not math.isfinite(args.min_probability) or not 0 <= args.min_probability <= 1:
        raise ValueError("min-probability 必须在 [0, 1] 内")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("输出目录不能与输入目录相同，也不能位于输入目录内部")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须不存在或为空，避免覆盖之前的结果")
    paths = image_paths(source)
    if not paths:
        raise ValueError("输入目录中没有支持的图片")

    # joblib uses pickle: only load artifacts produced by trusted training runs.
    artifact = joblib.load(args.model_dir.expanduser() / "classifier.joblib")
    config, classifier = artifact["config"], artifact["classifier"]
    if config["format_version"] != 1 or config["preprocessing"] != "exif_alpha_white_square_pad_v1":
        raise ValueError("不支持的模型版本或预处理配置")
    validate_classes(config["classes"])
    if sorted(classifier.classes_.tolist()) != list(range(len(config["classes"]))):
        raise ValueError("分类头与类别映射不一致")
    # Load once, reuse for every batch. Same CLIP preprocessing as train/predict.
    encoder = ClipEncoder(config["model"], args.pretrained or config["pretrained"],
                          args.device, args.batch_size)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "classification_results.jsonl"
    summary = {"total": len(paths), "classified": 0, "review": 0, "failed": 0,
               "output": str(output), "report": str(report_path)}
    counts = {}
    with report_path.open("x", encoding="utf-8") as report:
        def write_row(row):
            report.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            report.flush()

        for start in range(0, len(paths), args.batch_size):
            batch = paths[start:start + args.batch_size]
            try:
                results = predict_batch(encoder, classifier, config, batch, args)
                items = list(zip(batch, results, [None] * len(batch)))
            except Exception:
                if not args.skip_errors:
                    raise
                # One unreadable image should not discard the other images in its batch.
                # Retry individually only after a failed batch; successful batches stay fast.
                items = []
                for path in batch:
                    try:
                        result = predict_batch(encoder, classifier, config, [path], args)[0]
                        items.append((path, result, None))
                    except Exception as exc:
                        items.append((path, None, str(exc)))

            for path, result, error in items:
                if error is not None:
                    summary["failed"] += 1
                    write_row({"path": str(path), "status": "error", "error": error})
                    print(f"跳过：{path}：{error}", file=sys.stderr, flush=True)
                    continue
                folder = result["class_name"] if result["accepted"] else REVIEW_FOLDER
                # Preserve source-relative paths below each class. Duplicate basenames
                # from different source folders therefore cannot overwrite one another.
                destination = output / folder / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                shutil.copystat(path, destination)
                summary["classified" if result["accepted"] else "review"] += 1
                counts[folder] = counts.get(folder, 0) + 1
                write_row({"path": str(path), "output_path": str(destination),
                           "status": "copied", "probability": result["top_k"][0]["probability"], **result})
            print(f"已处理 {min(start + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    summary["folder_counts"] = counts
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="图片输入目录，递归读取")
    parser.add_argument("--output", "-o", type=Path, required=True, help="分类输出目录，需不存在或为空")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "output",
                        help="训练结果目录，默认当前脚本旁的 output")
    parser.add_argument("--pretrained", help="与训练时相同的 CLIP 本地权重文件")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-probability", type=float, default=0.0,
                        help="低于阈值的图片复制到 _待复核；默认全部按最佳类别归档")
    parser.add_argument("--skip-errors", action="store_true", help="跳过推理失败的图片，将错误写入结果清单")
    return parser.parse_args()


def main():
    try:
        summary = classify_directory(parse_args())
    except Exception as exc:
        print(f"分类失败：{exc}\n若已有部分输出，请查看 classification_results.jsonl，并用新的输出目录重试。", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Even with --skip-errors, report partial failure to automation callers.
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
