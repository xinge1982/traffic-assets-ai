import torch
from depth_anything_3.api import DepthAnything3

# $env:HTTP_PROXY="http://172.20.10.2:18080"
# $env:HTTPS_PROXY="http://172.20.10.2:18080"
# pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

class DepthAnythingV3:


    def __init__(self):

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        print(
            "Loading Depth Anything V3..."
        )


        self.device = device


        self.model = DepthAnything3.from_pretrained(
            "./models/depth_anything_v3"
        )


        self.model = self.model.to(
            device=device
        )


        self.model.eval()


        print(
            "Depth Anything V3 loaded"
        )



    def predict(
        self,
        image
    ):


        prediction = self.model.inference(
            [
                image
            ]
        )

        return prediction
