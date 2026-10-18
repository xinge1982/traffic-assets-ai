#!/usr/bin/env python3
"""重新组织目录中的标志牌文件：复制到输出目录，每个文件增加一层分类目录。

示例：
  输入目录/10100停车让行标志.png
  → 输出目录/10100_停车让行标志/10100停车让行标志.png

运行（Python 3.11，无第三方依赖）：
  python reorganize_sign_zip.py -i "D:/signs/input" -o "D:/signs/output"
  python reorganize_sign_zip.py -i "D:/signs/input" -o "D:/signs/output" --separator "-"

输入和输出都是目录，不读取或生成 ZIP。递归读取，保留相对于输入目录的
子目录结构；源文件保持不变。输出目录可以不存在或为空，非空时拒绝写入。
所有文件都必须以前五位数字开头；不符合规则时，在复制之前报告错误。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys


def reorganize(input_path: Path, output_path: Path, separator: str = "_") -> int:
    if not separator or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in separator):
        raise ValueError("分隔符不能为空，且不能包含路径分隔符或 Windows 文件名禁用字符")
    source = input_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"输入目录不存在或不是目录：{source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("输出目录不能与输入目录相同，也不能位于输入目录内部")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"输出目录已存在且非空，或不是目录：{output}")

    # 先检查全部文件，命名不符合要求时不创建输出文件。
    plan = []
    seen = set()
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"不支持符号链接：{path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"不支持的文件类型：{path}")
        relative = path.relative_to(source)
        match = re.fullmatch(r"([0-9]{5})(.*)", path.stem)
        if not match:
            raise ValueError(f"文件名不是以五位数字开头：{relative}")
        code, name = match.groups()
        folder = code + separator + name if name else code
        destination = relative.parent / folder / path.name
        key = destination.as_posix().casefold()
        if key in seen:
            raise ValueError(f"生成的文件路径重复：{destination}")
        seen.add(key)
        plan.append((path, destination))
    if not plan:
        raise ValueError("输入目录中没有可处理的文件")

    output.mkdir(parents=True, exist_ok=True)
    for index, (original, relative_target) in enumerate(plan, 1):
        target = output / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        # xb 模式防止覆盖已有文件；不解码图片，直接复制原始字节。
        with original.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        shutil.copystat(original, target)
        print(f"[{index}/{len(plan)}] {relative_target}")
    return len(plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=Path, help="输入目录，递归读取文件")
    parser.add_argument("-o", "--output", type=Path, help="输出目录；默认：输入目录名_分类（同级目录）")
    parser.add_argument("--separator", default="_", help="五位数字与名称之间的分隔符，默认下划线")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output or source.with_name(source.name + "_分类")
    try:
        count = reorganize(source, output, args.separator)
    except (OSError, ValueError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1
    print(f"\n处理完成：{count} 个文件\n输出：{output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
