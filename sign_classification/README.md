# 交通标志牌 CLIP 分类

按 `类别名称/图片` 读取标志牌渲染图，用预训练 CLIP 图像编码器提取 L2 归一化特征，再训练逻辑回归分类头（linear probe）。本版本冻结 CLIP 主干，仅训练分类头，不是从零训练或微调 CLIP。类别名不参与文本编码，支持中文名称和业务编码；新增类别后需要重新训练。

## 安装

建议 Python 3.11，以下命令在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -r sign_classification/requirements.txt
```

GPU 使用与本机硬件匹配的 [PyTorch 官方安装命令](https://pytorch.org/get-started/locally/) 安装 torch/torchvision，再安装其余依赖。`--device auto` 有 CUDA 时使用 GPU，否则使用 CPU；分类头在 CPU 上训练。显存不足时减小 batch-size。

## 数据组织

```text
data/train/
├── 禁止左转/
│   ├── 001.png
│   └── 002.png
├── 禁止右转/
│   ├── 001.png
│   └── 002.png
└── 限速60/
    ├── 001.png
    └── 002.png
```

一级目录名是类别，类别内递归读取 JPG/JPEG/PNG/BMP/WebP/TIF/TIFF。至少两个类别；空类别、坏图报错。类别排序后分配索引，并保存至模型，预测不重新扫描类别目录。

输入应为单个标志牌渲染图或裁剪图，本模块不检测整张道路图片中的标志牌位置。建议包含不同视角、背景、光照。透明图合成白色背景，长方形图补白成正方形，再使用 CLIP 推理预处理，保留完整牌面。训练和预测预处理一致，不做水平翻转，以免改变方向含义。

## 训练

```bash
python -m sign_classification.train --data-dir D:/signs/train --output-dir sign_classification/output --batch-size 16 --val-ratio 0.2
```

默认 `ViT-B-32` + `laion2b_s34b_b79k`，首次运行需下载权重。离线时可指定对应的本地权重：

```bash
python -m sign_classification.train --data-dir D:/signs/train --model ViT-B-32 --pretrained D:/models/open_clip_pytorch_model.bin --device cuda --batch-size 8
```

默认按类别随机划分约 20% 验证集，每类至少两张，保证训练/验证均保留图片。只有一张时可用 `--val-ratio 0`，但此时无验证指标。`--seed 42` 控制划分，`--C 10` 为逆正则化系数，`--max-iter 1000` 为最大迭代次数；收敛警告时可增加迭代次数。分类头使用 balanced 类别权重。

同一模型的不同视角渲染图高度相似，建议按源模型或物理标志牌分组，手工组织独立验证集：

```bash
python -m sign_classification.train --data-dir D:/signs/train --val-dir D:/signs/val --output-dir sign_classification/output
```

验证集类别目录须与训练集一致；指定 val-dir 后不再随机划分。代码拒绝跨训练/验证集合的字节相同图片，以及同一图片标注不同类别。近似图片和同一源模型的不同视角仍需自行分组。另行保留独立测试集，避免反复调参影响评价。

| 输出文件 | 内容 |
| --- | --- |
| classifier.joblib | 分类头及推理必需配置、类别映射 |
| classes.json | 类别名到索引的映射 |
| config.json | CLIP 型号、权重来源、预处理版本、特征维数、训练参数 |
| features.npz | features、labels、train_indices、val_indices |
| samples.json | 按特征行顺序记录图片绝对路径、类别、SHA256、数据划分 |
| metrics.json | 训练准确率、验证准确率、每类 precision/recall/F1、混淆矩阵 |

特征为 float32；混淆矩阵行是真实类别、列是预测类别，顺序对应 config.json 的 classes。无验证集时 validation=null。输出不重复打包 CLIP 主干权重，部署仍需相同权重。再次训练覆盖同名文件，保留版本时请指定不同 output-dir。模块的 data/、output/ 已在 .gitignore 排除。

## 识别

```bash
# 单图
python -m sign_classification.predict --model-dir sign_classification/output --input D:/signs/test.png --top-k 3
# 目录递归批量识别，保存 JSON
python -m sign_classification.predict --model-dir sign_classification/output --input D:/signs/test --output sign_classification/output/predictions.json --top-k 3 --min-probability 0.6
```

输出 path、最佳 class_name、accepted、top_k（类别和 probability）。最佳概率低于阈值时 accepted=false，仍保留候选供人工复核。概率未经校准，仅表示已知类别间的分类分数，阈值不能保证检测未知类别。数字等细节相近的类别应检查混淆矩阵和错误样本。

迁移机器后可加 `--pretrained D:/models/open_clip_pytorch_model.bin`，必须是训练时完全相同的权重，维数相同不代表兼容。请将模型目录和原始 CLIP 权重一起管理。只加载可信来源的 classifier.joblib，其底层 pickle 格式可以执行代码。

## 按预测类别批量整理图片

`classify_directory.py` 递归扫描输入目录，读取训练结果，按预测类别**复制**图片。源文件不移动、不修改。默认读取脚本旁的 `sign_classification/output/classifier.joblib`，模型只加载一次；CLIP 特征按批次处理，避免一次保存全部特征。

在仓库根目录运行：

```powershell
.\vent\Scripts\python.exe -m sign_classification.classify_directory `
  --input "D:/signs/unclassified" `
  --output "D:/signs/classified" `
  --pretrained "D:/models/open_clip_pytorch_model.bin" `
  --batch-size 16
```

`--pretrained` 使用训练时相同的本地 CLIP 权重；如果训练配置中记录的权重路径仍有效，可以省略。自定义训练结果目录用 `--model-dir D:/models/sign_classifier`。脚本也支持直接执行，便于 PyCharm 使用“脚本路径”运行：

```powershell
.\vent\Scripts\python.exe sign_classification/classify_directory.py --input "D:/signs/unclassified" --output "D:/signs/classified"
```

PyCharm 的参数填写 `--input "D:/signs/unclassified" --output "D:/signs/classified" --pretrained "D:/models/open_clip_pytorch_model.bin"`；也可使用模块名称 `sign_classification.classify_directory`。相对输入、输出路径基于工作目录解析；默认模型目录基于脚本位置解析。

| 输入示例 | 输出示例（假设预测为 10100_停车让行标志） |
| --- | --- |
| unclassified/a.png | classified/10100_停车让行标志/a.png |
| unclassified/camera1/a.png | classified/10100_停车让行标志/camera1/a.png |
| unclassified/camera2/a.png | classified/10100_停车让行标志/camera2/a.png |

类别目录名来自训练配置，不根据图片文件名推断。保留类别目录下的输入相对路径以避免同名图片覆盖。输出目录必须不存在或为空，且不能放在输入目录内部。支持的图片格式与训练一致，非图片文件忽略。

- `--device auto/cpu/cuda/cuda:0`：选择推理设备，默认 auto。
- `--top-k 3`：结果清单保留的候选数，默认 3。
- `--min-probability 0.6`：可选阈值；最佳概率低于阈值的图片放入 `_待复核`。默认 0，全部按最佳类别归档。阈值不能保证检测未知类别。
- `--skip-errors`：遇到推理失败的批次时逐张重试，跳过失败图片并记录错误。默认遇错停止；文件写入错误始终停止。任何图片推理失败时，即使启用跳过，最终退出码仍为 1。

输出根目录另有 `classification_results.jsonl`（每行一张图片的源路径、输出路径、预测类别、最佳概率、accepted、Top-K 或错误）和 `summary.json`（处理数量和各输出目录计数）。中途失败时已复制的图片及已写入的清单保留，`summary.json` 只在全部处理完成后生成；重试请使用新的输出目录。

同时生成 `classification_results.csv`，使用 UTF-8 BOM 编码，方便 Excel 打开中文。列为 `pic`（原图片文件名，含扩展名）、`class`（模型预测的最佳类别）和 `confidence`（该类别的概率，范围 0～1）。低置信度图片仍记录预测类别，不写成 `_待复核`；启用 `--skip-errors` 时失败图片也有一行，但类别和置信度留空，具体错误见 JSONL。同名图片可能有多行，其完整路径可在 JSONL 中查看。CSV 逐行保存，中途停止时保留已写入结果。

## 将人工核对后的目录重新导出为 CSV

把图片人工移动到正确类别目录后，使用 `export_reviewed_csv.py` 根据**实际目录**重新生成结果，不加载模型，不读取旧 CSV 的预测类别。该脚本仅使用 Python 标准库。

```powershell
.\vent\Scripts\python.exe -m sign_classification.export_reviewed_csv --input "D:/signs/reviewed"
# 也支持 PyCharm 直接运行文件：
.\vent\Scripts\python.exe sign_classification/export_reviewed_csv.py --input "D:/signs/reviewed"
```

输入目录下的一级子目录作为类别，递归收集其内部的 JPG/JPEG/PNG/BMP/WebP/TIF/TIFF/GIF。比如 `reviewed/10100_停车让行标志/camera1/001.png` 导出为 `001.png,10100_停车让行标志,1.0`。输入目录根部的文件不属于任何类别，不参与导出；空类别目录不产生记录。每个一级子目录都按类别处理，包括 `_待复核`，如不需要它，请先将其移出输入目录。

在输入目录中生成或更新 `classification_results.csv`，列为 `pic,class,confidence`：图片原文件名（含扩展名）、类别目录名和固定值 `1.0`。使用 UTF-8 BOM 编码，支持 Excel 中文。`confidence=1.0` 用于标记人工确认结果并兼容原有三列读取程序，不代表模型计算出的概率。图片文件保持不变，CSV 写入成功后替换旧文件。没有图片时报告错误并保留旧 CSV。同名图片全部保留并在命令行提醒，以 pic 为键转换为 map 时应先处理重名。

## 对比旧分类和人工核对后的目录

`compare_classification_dirs.py` 读取两边的实际图片目录，以**文件名（含扩展名，区分大小写）**匹配图片，比较它们所在的一级类别目录。不加载模型，也不读取已有结果 CSV；类别内子目录的移动不算类别变更。支持的图片格式、根目录文件忽略规则与人工核对 CSV 导出一致。

```powershell
.\vent\Scripts\python.exe -m sign_classification.compare_classification_dirs `
  --old-dir "D:/signs/classified_old" `
  --reviewed-dir "D:/signs/reviewed" `
  --output "D:/signs/classification_diff.csv"
```

也可在 PyCharm 中直接运行 `sign_classification/compare_classification_dirs.py`，传入相同参数。省略 `--output` 时生成 `核对目录/classification_diff.csv`；已有输出 CSV 会被更新。UTF-8 BOM 编码方便 Excel 打开中文。

CSV 列为 `pic,old_class,new_class,status,old_path,new_path`。路径相对于各自输入根目录。默认只输出差异，加 `--include-unchanged` 则导出全部记录。

| status | 含义 |
| --- | --- |
| class_changed | 两边图片文件名相同，但类别目录不同 |
| only_old | 只在旧目录找到该图片文件名 |
| only_new | 只在核对目录找到该图片文件名 |
| ambiguous | 任一目录中同名图片超过一张，无法唯一匹配；类别和路径用 ` | ` 列出，不猜测对应关系 |
| unchanged | 类别不变，仅在指定 `--include-unchanged` 时写入 CSV |

本脚本只比较按文件名匹配的类别归属，不比较图片内容。图片被重命名时会分别表现为 only_old 和 only_new，不会自动判断它们是同一张图。

## 检查

```bash
python -m unittest discover -s sign_classification/tests -v
```

测试用固定特征编码器代替 CLIP 下载，验证数据扫描、划分、预处理、真实分类头训练、保存加载、Top-K 和重复检测。真实识别效果需要安装完整依赖后以实际训练集验证。

接口参考：[OpenCLIP 官方文档](https://github.com/mlfoundations/open_clip)。

## Go 本地推理

已提供 [ONNX 导出脚本](export_onnx.py) 和 [Go 推理模块](go/README.md)。训练完成后一次性导出 CLIP 图像编码器、特征归一化和分类头；Go 部署时仅需导出的模型和 ONNX Runtime，无需 Python。当前导出支持本项目默认的 `ViT-B-32`。

```bash
# 仓库根目录，已有 output/classifier.joblib 后执行
python -m pip install -r sign_classification/requirements-export.txt
python -m sign_classification.export_onnx --model-dir sign_classification/output --verify-image D:/signs/test.png
```

Windows 编译、单图/目录预测、Go 调用示例和概率核对见 [go/README.md](go/README.md)。
