"""Export the trained frozen CLIP + logistic head for standalone Go inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def head_parameters(classifier, classes, feature_dim):
    """Keep sklearn probability column ordering, including binary sigmoid."""
    indices = [int(i) for i in classifier.classes_]
    if sorted(indices) != list(range(len(classes))):
        raise ValueError("分类头类别索引与 classes 配置不一致")
    weights = np.asarray(classifier.coef_, dtype=np.float32)
    bias = np.asarray(classifier.intercept_, dtype=np.float32)
    expected_rows = 1 if len(classes) == 2 else len(classes)
    if weights.shape != (expected_rows, feature_dim) or bias.shape != (expected_rows,):
        raise ValueError("分类头参数维数错误")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("分类头含非有限数值")
    return weights, bias, [classes[i] for i in indices]


def export(args):
    import onnx
    import onnxruntime as ort
    import torch
    from open_clip import get_model_preprocess_cfg

    from .common import ClipEncoder, load_image

    artifact = joblib.load(args.model_dir / "classifier.joblib")
    config, classifier = artifact["config"], artifact["classifier"]
    if config["format_version"] != 1 or config["preprocessing"] != "exif_alpha_white_square_pad_v1":
        raise ValueError("不支持的训练模型版本或预处理")
    # Other OpenCLIP models may use a different resize/crop or image architecture.
    if config["model"] != "ViT-B-32":
        raise ValueError("当前 Go 导出仅支持 ViT-B-32，请使用该型号训练")
    weights, bias, classes = head_parameters(classifier, config["classes"], config["feature_dim"])
    encoder = ClipEncoder(config["model"], args.pretrained or config["pretrained"], "cpu", 1)
    pp = get_model_preprocess_cfg(encoder.model)
    size = pp["size"]
    if isinstance(size, int):
        size = (size, size)
    if (tuple(size) != (224, 224) or pp.get("interpolation", "bicubic") != "bicubic"
            or pp.get("resize_mode", "shortest") != "shortest"):
        raise ValueError(f"Go 尚不支持该预处理配置: {pp}")

    class CombinedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = encoder.model.visual
            self.register_buffer("weight", torch.from_numpy(weights))
            self.register_buffer("bias", torch.from_numpy(bias))

        def forward(self, images):
            features = torch.nn.functional.normalize(self.visual(images).float(), dim=-1)
            logits = torch.nn.functional.linear(features, self.weight, self.bias)
            if len(classes) == 2:
                positive = torch.sigmoid(logits)
                return torch.cat((1 - positive, positive), dim=1)
            return torch.softmax(logits, dim=1)

    model = CombinedModel().eval()
    # Force the conventional attention path for the legacy ONNX exporter.
    torch.backends.mha.set_fastpath_enabled(False)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    # Never leave a prior successful manifest paired with an incomplete export.
    manifest_path.unlink(missing_ok=True)
    onnx_path = output / "sign_classifier.onnx"
    torch.manual_seed(42)
    sample = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(onnx_path), input_names=["images"],
                          output_names=["probabilities"], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    references = []
    samples = [(None, sample)]
    for path in args.verify_image:
        samples.append((path, encoder.preprocess(load_image(path)).unsqueeze(0)))
    for path, tensor in samples:
        with torch.inference_mode():
            features = torch.nn.functional.normalize(encoder.model.encode_image(tensor).float(), dim=-1)
        expected = classifier.predict_proba(features.numpy())
        actual = session.run(["probabilities"], {"images": tensor.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        if path:
            references.append({"path": str(path.resolve()), "probabilities": expected[0].tolist()})
    manifest = {
        "format_version": 1, "model_file": onnx_path.name,
        "input_name": "images", "output_name": "probabilities", "image_size": 224,
        "preprocessing": "exif_alpha_white_square_pad_bicubic_v1",
        "mean": list(pp["mean"]), "std": list(pp["std"]), "classes": classes,
        "source_model": config["model"], "source_pretrained": config["pretrained"],
        "references": references,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出并通过 ONNX/Python 概率对比: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("sign_classification/output"))
    parser.add_argument("--output-dir", type=Path, default=Path("sign_classification/output/onnx"))
    parser.add_argument("--pretrained", help="训练时相同的 CLIP 本地权重")
    parser.add_argument("--verify-image", type=Path, action="append", default=[],
                        help="可重复指定；保存 Python 参考概率供 Go 核对")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
