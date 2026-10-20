"""根据人工核对目录校正训练集类别，只移动训练集中已存在且可唯一匹配的图片。

按文件名（含扩展名）匹配；两边一级子目录作为类别，类别目录内递归读取。
不复制 reviewed 中的新图片，不覆盖目标图片，不修改 reviewed 目录。
可先用 --dry-run 预览；默认实际移动，并生成 CSV 和 JSON 操作汇总。
Python 3.11，无第三方依赖。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

if __package__:
    from .compare_classification_dirs import scan_classes
else:
    from compare_classification_dirs import scan_classes


FIELDS = ["pic", "old_class", "new_class", "status", "source_path", "target_path",
          "reviewed_path", "message"]


def move_no_overwrite(source: Path, target: Path):
    """Create the target exclusively; delete the original only after a full copy."""
    created = False
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            created = True
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        shutil.copystat(source, target)
        source.unlink()
    except BaseException:
        # An interrupt can arrive immediately after unlink succeeded. Never
        # delete the destination when the original no longer exists.
        if created and source.exists():
            target.unlink(missing_ok=True)
        raise


def reconcile_training_classes(train_dir: Path, reviewed_dir: Path,
                               report: Path | None = None, dry_run: bool = False) -> dict:
    training = train_dir.expanduser().resolve()
    reviewed = reviewed_dir.expanduser().resolve()
    if training == reviewed or training.is_relative_to(reviewed) or reviewed.is_relative_to(training):
        raise ValueError("训练目录与核对目录不能相同，也不能互相嵌套")
    train_index, review_index = scan_classes(training), scan_classes(reviewed)
    if not review_index:
        raise ValueError("人工核对目录中没有可处理的图片")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report = (report or training / f"training_reconcile_{stamp}.csv").expanduser().resolve()
    if report.suffix.lower() != ".csv":
        raise ValueError("操作报告必须使用 .csv 扩展名")
    summary_path = report.with_suffix(".summary.json")
    # Reports must also stay outside reviewed: do not change that directory.
    if report.is_relative_to(reviewed):
        raise ValueError("报告不能写入人工核对目录内部")
    if report.exists() or summary_path.exists():
        raise FileExistsError("报告或汇总已存在，请指定新的 --report 路径")
    report.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    moved_between = Counter()
    with report.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        for pic in sorted(train_index.keys() | review_index.keys()):
            old = train_index.get(pic, [])
            new = review_index.get(pic, [])
            row = {"pic": pic, "old_class": " | ".join(sorted({c for c, _ in old})),
                   "new_class": " | ".join(sorted({c for c, _ in new})),
                   "source_path": " | ".join(str(training / p) for _, p in old),
                   "target_path": "", "reviewed_path": " | ".join(str(reviewed / p) for _, p in new),
                   "message": ""}
            if not new:
                row["status"] = "not_reviewed"
            elif len(old) > 1 or len(new) > 1:
                row.update(status="ambiguous", message="同名图片无法唯一匹配，未移动")
            elif not old:
                row["status"] = "not_in_training"
            elif old[0][0] == new[0][0]:
                row["status"] = "unchanged"
            else:
                original = training / old[0][1]
                # Retain any camera/subfolder path below the old class.
                target = training / new[0][0] / Path(*Path(old[0][1]).parts[1:])
                row["target_path"] = str(target)
                try:
                    if target.exists() or target.is_symlink():
                        row.update(status="conflict", message="目标已存在，未覆盖、未移动")
                    elif not target.resolve().is_relative_to(training):
                        row.update(status="conflict", message="目标路径指向训练目录之外，未移动")
                    elif dry_run:
                        row["status"] = "would_move"
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        move_no_overwrite(original, target)
                        row["status"] = "moved"
                        moved_between[(old[0][0], new[0][0])] += 1
                except FileExistsError:
                    row.update(status="conflict", message="目标已存在，未覆盖、未移动")
                except OSError as exc:
                    row.update(status="error", message=str(exc))
            counts[row["status"]] += 1
            writer.writerow(row)
            stream.flush()

    result = {
        "training_dir": str(training), "reviewed_dir": str(reviewed), "dry_run": dry_run,
        "training_images": sum(map(len, train_index.values())),
        "reviewed_images": sum(map(len, review_index.values())), "counts": dict(counts),
        "moves_by_class": [{"old_class": old, "new_class": new, "count": count}
                           for (old, new), count in sorted(moved_between.items())],
        "report": str(report), "summary": str(summary_path),
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True, help="训练图片根目录（类别目录的上一级）")
    parser.add_argument("--reviewed-dir", type=Path, required=True, help="人工核对后的分类根目录")
    parser.add_argument("--report", type=Path, help="CSV 报告路径，默认训练目录下带时间戳的文件")
    parser.add_argument("--dry-run", action="store_true", help="只输出预览报告，不移动图片")
    args = parser.parse_args()
    try:
        result = reconcile_training_classes(args.train_dir, args.reviewed_dir, args.report, args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"校正失败：{exc}", file=sys.stderr)
        return 1
    labels = {"moved": "已移动", "would_move": "预计移动", "unchanged": "类别一致",
              "not_reviewed": "训练图片未出现在核对目录", "not_in_training": "核对图片未出现在训练集",
              "ambiguous": "同名歧义，跳过", "conflict": "目标冲突，跳过", "error": "操作失败"}
    for status, label in labels.items():
        print(f"{label}: {result['counts'].get(status, 0)}")
    print(f"CSV：{result['report']}\n汇总：{result['summary']}")
    return 1 if result["counts"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
