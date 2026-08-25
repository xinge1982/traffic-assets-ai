from pathlib import Path

import torch
from depth_anything_3.api import DepthAnything3

# $env:HTTP_PROXY="http://172.20.10.2:18080"
# $env:HTTPS_PROXY="http://172.20.10.2:18080"
# pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"


class DepthAnythingV3:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading Depth Anything V3...")

        self.device = device

        # Resolve the model path relative to this Python file instead of the
        # current working directory. This keeps imports working when this
        # module is called from scripts in other directories.
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "models" / "depth_anything_v3"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Depth Anything V3 model directory not found: {model_path}"
            )

        self.model_path = model_path
        self.model = DepthAnything3.from_pretrained(str(model_path))
        self.model = self.model.to(device=device)
        self.model.eval()

        print(f"Depth Anything V3 loaded from: {model_path}")

    def predict(self, image):
        prediction = self.model.inference([image])
        return prediction
