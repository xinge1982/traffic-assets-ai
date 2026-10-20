"""比较旧分类目录与人工核对目录，按图片文件名匹配并导出类别差异 CSV。

一级子目录名作为类别，递归读取类别目录中的图片，忽略根目录文件。
默认只输出差异；--include-unchanged 可输出全部图片的对比记录。
不调用模型，不移动或修改图片。Python 3.11，无第三方依赖。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import sys
import tempfile

if __package__:
    from .export_reviewed_csv import IMAGE_EXTENSIONS
else:
    from export_reviewed_csv import IMAGE_EXTENSIONS


FIELDS = ["pic", "old_class", "new_class", "status", "old_path", "new_path"]


def scan_classes(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"分类根目录不存在或不是目录：{root}")
    index = defaultdict(list)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index[path.name].append((folder.name, path.relative_to(root).as_posix()))
    return index


def compare_directories(old_dir: Path, reviewed_dir: Path, output: Path | None = None,
                        include_unchanged: bool = False) -> dict:
    old_root = old_dir.expanduser().resolve()
    new_root = reviewed_dir.expanduser().resolve()
    old_index, new_index = scan_classes(old_root), scan_classes(new_root)
    output = (output or new_root / "classification_diff.csv").expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("输出文件必须使用 .csv 扩展名")

    rows = []
    counts = Counter()
    for pic in sorted(old_index.keys() | new_index.keys()):
        old = old_index.get(pic, [])
        new = new_index.get(pic, [])
        # Never infer pairings when the requested pic key is not unique.
        if len(old) > 1 or len(new) > 1:
            status = "ambiguous"
        elif not old:
            status = "only_new"
        elif not new:
            status = "only_old"
        elif old[0][0] != new[0][0]:
            status = "class_changed"
        else:
            status = "unchanged"
        counts[status] += 1
        if status == "unchanged" and not include_unchanged:
            continue
        rows.append({
            "pic": pic,
            "old_class": " | ".join(sorted({c for c, _ in old})),
            "new_class": " | ".join(sorted({c for c, _ in new})),
            "status": status,
            "old_path": " | ".join(p for _, p in old),
            "new_path": " | ".join(p for _, p in new),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=output.parent, prefix=".classification_diff_",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {"old_images": sum(map(len, old_index.values())),
            "new_images": sum(map(len, new_index.values())), "counts": dict(counts),
            "exported_rows": len(rows), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-dir", type=Path, required=True, help="旧分类结果的根目录")
    parser.add_argument("--reviewed-dir", type=Path, required=True, help="人工核对后的分类根目录")
    parser.add_argument("--output", "-o", type=Path, help="输出 CSV，默认：核对目录/classification_diff.csv")
    parser.add_argument("--include-unchanged", action="store_true", help="同时输出分类不变的图片")
    args = parser.parse_args()
    try:
        result = compare_directories(args.old_dir, args.reviewed_dir, args.output, args.include_unchanged)
    except (OSError, ValueError) as exc:
        print(f"对比失败：{exc}", file=sys.stderr)
        return 1
    print(f"旧目录 {result['old_images']} 张图片；核对目录 {result['new_images']} 张图片")
    names = {"class_changed": "分类变更", "only_old": "仅旧目录存在", "only_new": "仅核对目录存在",
             "ambiguous": "同名歧义", "unchanged": "分类不变"}
    for status, label in names.items():
        print(f"{label}: {result['counts'].get(status, 0)}")
    print(f"已导出 {result['exported_rows']} 条记录：{result['output']}")
    if result["counts"].get("ambiguous"):
        print("注意：同名图片无法唯一匹配，已标记为 ambiguous；请根据 CSV 中的路径人工确认。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
