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

## 检查

```bash
python -m unittest discover -s sign_classification/tests -v
```

测试用固定特征编码器代替 CLIP 下载，验证数据扫描、划分、预处理、真实分类头训练、保存加载、Top-K 和重复检测。真实识别效果需要安装完整依赖后以实际训练集验证。

接口参考：[OpenCLIP 官方文档](https://github.com/mlfoundations/open_clip)。
