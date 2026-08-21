import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Segmentor:


    def __init__(
        self,
        checkpoint,
        config,
        device="cuda"
    ):


        model = build_sam2(
            config_file=config,
            ckpt_path=checkpoint,
            device=device
        )


        self.predictor = SAM2ImagePredictor(
            model
        )



    def set_image(
        self,
        image_rgb
    ):

        self.predictor.set_image(
            image_rgb
        )



    def segment_box(
        self,
        bbox
    ):


        box=np.array(
            bbox
        )


        masks, scores, logits = (
            self.predictor.predict(
                box=box,
                multimask_output=True
            )
        )


        # choose largest mask

        best_mask=None
        best_area=0


        for mask in masks:

            area=np.sum(mask)


            if area > best_area:

                best_area=area
                best_mask=mask



        return {
            "mask":best_mask,
            "score":float(
                max(scores)
            )
        }