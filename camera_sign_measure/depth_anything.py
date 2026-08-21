import torch
from depth_anything_3.api import DepthAnything3


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
