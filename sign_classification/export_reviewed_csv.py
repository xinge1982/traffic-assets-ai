"""将人工核对后的类别目录导出为输入目录中的 classification_results.csv。

输入目录的一级子目录名作为 class，递归读取各类别目录中的图片。
pic 为原图片文件名（含扩展名）。导出 pic,class,confidence 三列，confidence 固定为 1.0；不推理、不移动图片。
输出使用 UTF-8 BOM，方便 Excel 打开中文；已有同名 CSV 会被更新。

示例（Python 3.11，无第三方依赖）：
  python sign_classification/export_reviewed_csv.py --input D:/signs/reviewed
  python -m sign_classification.export_reviewed_csv --input D:/signs/reviewed
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def export_reviewed_csv(input_dir: Path) -> dict:
    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"输入目录不存在或不是目录：{root}")

    rows = []
    counts = {}
    seen = set()
    duplicate_count = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        count = 0
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append((path.name, folder.name, 1.0))
            # Keep every image, but report possible ambiguity for pic-keyed maps.
            key = path.name.casefold()
            if key in seen:
                duplicate_count += 1
            seen.add(key)
            count += 1
        if count:
            counts[folder.name] = count
    if not rows:
        raise ValueError("各类别子目录中没有找到图片；输入目录下应直接包含各个类别目录")

    output = root / "classification_results.csv"
    temporary = None
    try:
        # Finish writing and close the temporary file before replacing, including
        # on Windows. A scan/write failure leaves the previous CSV unchanged.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=root, prefix=".classification_results_", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(["pic", "class", "confidence"])
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {"output": str(output), "images": len(rows), "classes": len(counts),
            "folder_counts": counts, "duplicate_names": duplicate_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True, help="人工核对后的分类根目录")
    args = parser.parse_args()
    try:
        result = export_reviewed_csv(args.input)
    except (OSError, ValueError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 1
    for name, count in result["folder_counts"].items():
        print(f"{name}: {count} 张")
    print(f"完成：{result['classes']} 个类别，{result['images']} 张图片")
    print(f"输出：{result['output']}")
    if result["duplicate_names"]:
        print(f"注意：有 {result['duplicate_names']} 条同名图片记录，CSV 全部保留；"
              "以 pic 转换为 map 时请处理同名情况。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
