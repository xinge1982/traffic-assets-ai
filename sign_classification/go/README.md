# Go 标志牌分类推理

Go 读取图片 → 白底合成和正方形补白 → bicubic 缩放和 RGB CHW 归一化 → ONNX Runtime → Top-K 类别。

ONNX 内包含 CLIP 图像编码器、L2 特征归一化、逻辑回归头和概率计算。两类使用 sigmoid，多类使用 softmax，类别顺序取训练分类头。支持中文类别、单图、目录递归批量识别、概率阈值和 JSON 输出。输入应为单个标志牌的渲染图或裁剪图。

## 1. 导出已有训练结果

在仓库根目录执行，训练仍使用原来的 Python 代码：

```bash
python -m pip install -r sign_classification/requirements-export.txt
python -m sign_classification.export_onnx --model-dir sign_classification/output --output-dir sign_classification/output/onnx --verify-image D:/signs/test.png
```

首次导出需取得训练时相同的 CLIP 权重，也可加 `--pretrained D:/models/open_clip_pytorch_model.bin`。目前仅支持 `ViT-B-32` 的 224×224、bicubic、shortest 预处理配置；其他配置明确报错。只加载可信的 classifier.joblib。

导出生成：

| 文件 | 用途 |
| --- | --- |
| sign_classifier.onnx | 完整图像编码器与分类头，固定单张输入 `[1,3,224,224]` |
| manifest.json | 输入输出名称、预处理、类别顺序、可选的参考图片概率 |

导出会用 ONNX checker 检查模型，并以 CPU ONNX Runtime 与 Python 的分类概率比较；失败时不写成功清单。可重复传入 `--verify-image` 增加真实图片检查。完整模型含 CLIP 主干，体积为数百 MB，Go 推理时不再下载 CLIP 权重。将这两个文件作为同一个模型版本部署。

## 2. 准备 Go 与 ONNX Runtime

需要 Go 1.23+、CGO 和 C 编译器。Windows 可使用 MinGW-w64 GCC，并确保 `gcc` 在 PATH 中。当前 go.mod 固定 `onnxruntime_go v1.22.0`，请使用 [ONNX Runtime v1.22.0 官方发行包](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0)。

- Windows x64：解压 `onnxruntime-win-x64-1.22.0.zip`，使用 `lib/onnxruntime.dll`，安装所需的 Microsoft Visual C++ Runtime。
- Linux x64：解压 `onnxruntime-linux-x64-1.22.0.tgz`，使用 `lib/libonnxruntime.so.1.22.0`。
- Go、C 编译器和 Runtime 的 CPU 架构应一致。当前默认 CPU 推理，未配置 CUDA provider。

## 3. 命令行识别

在 `sign_classification/go` 目录执行，Windows PowerShell：

```powershell
$env:CGO_ENABLED = "1"
go mod download
go build -o sign-predict.exe ./cmd/sign-predict

.\sign-predict.exe -ort-lib D:/models/onnxruntime/lib/onnxruntime.dll -model-dir ../output/onnx -input D:/signs/test.png -top-k 3

# 递归批量识别；结果写 JSON。output 的父目录应已存在。
.\sign-predict.exe -ort-lib D:/models/onnxruntime/lib/onnxruntime.dll -model-dir ../output/onnx -input D:/signs/test -top-k 3 -min-probability 0.6 -output predictions.json
```

Linux：

```bash
CGO_ENABLED=1 go build -o sign-predict ./cmd/sign-predict
./sign-predict -ort-lib /opt/onnxruntime/lib/libonnxruntime.so.1.22.0 -model-dir ../output/onnx -input /data/sign.png -top-k 3
```

输出格式与 Python 预测一致：

```json
[
  {
    "path": "D:/signs/test.png",
    "class_name": "禁止左转",
    "accepted": true,
    "top_k": [
      {"class_name": "禁止左转", "probability": 0.92},
      {"class_name": "禁止右转", "probability": 0.08}
    ]
  }
]
```

支持 JPG/JPEG、PNG、BMP、GIF、TIFF、WebP；动图仅读取第一帧。遇到坏图或 Runtime 错误时返回非零退出码，不输出部分批处理结果。概率未经校准，`accepted=false` 表示低于阈值，不能保证排除未知类别。

## 4. 集成到现有 Go 程序

可以复制本目录的包文件到你的项目并安装 go.mod 中的依赖，或将本目录作为模块导入。它位于仓库子目录，未单独发布版本标签；从远程依赖时可指定该功能的提交 SHA：

```bash
go get github.com/xinge1982/traffic-assets-ai/sign_classification/go@<commit-sha>
```

```go
package main

import (
    "fmt"
    "log"

    signs "github.com/xinge1982/traffic-assets-ai/sign_classification/go"
    ort "github.com/yalue/onnxruntime_go"
)

func main() {
    ort.SetSharedLibraryPath("D:/models/onnxruntime/lib/onnxruntime.dll")
    if err := ort.InitializeEnvironment(); err != nil { log.Fatal(err) }
    defer ort.DestroyEnvironment()

    classifier, err := signs.New("D:/models/sign_classifier")
    if err != nil { log.Fatal(err) }
    defer classifier.Close()

    result, err := classifier.Predict("D:/signs/test.png", 3, 0.6)
    if err != nil { log.Print(err); return }
    fmt.Println(result.ClassName, result.Accepted, result.TopK)
}
```

Runtime 在进程启动时初始化一次；先关闭所有分类器，再销毁 Runtime。同一个分类器可反复使用，内部串行保护推理与关闭；批量命令逐张复用同一模型。批量较大时通过包接口逐条保存结果，避免在内存中积累所有结果。

## 5. 核对 Go / Python 结果

```powershell
# 使用导出时 --verify-image 保存的参考图片及 Python 概率
.\sign-predict.exe -ort-lib D:/models/onnxruntime/lib/onnxruntime.dll -model-dir ../output/onnx -verify -tolerance 0.02
```

参考图片路径为导出机器的绝对路径，核对时必须仍可访问。Go 对 JPEG 自动按 EXIF 方向旋转、透明合成白底、居中补白并执行 a=-0.5 的 bicubic 抗锯齿缩放。Pillow 与 Go 的解码和缩放取整仍可能有数值差异，`-verify` 会比较所有类别的概率，超过容差则失败。0.02 是可配置的概率绝对误差上限，并非精度保证，建议使用真实验证集检查混淆类别。带方向元数据的非 JPEG 图片请先转成方向正确的 PNG。

```bash
go test ./...
go vet ./...
# 设置后额外运行真实 ONNX Runtime 集成测试（tiny 为合成常量模型）
ONNXRUNTIME_SHARED_LIBRARY_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.22.0 go test ./... -v
```

PNG 测试图片和 Pillow 参考结果覆盖透明合成、上采样和下采样；测试中的合成模型仅验证调用流程。实际 CLIP 分类效果仍需使用你的训练权重及独立验证集评估。
