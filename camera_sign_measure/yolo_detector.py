from ultralytics import YOLO

class YOLODetector:

    def __init__(
        self,
        model_path="models/yolov8l-worldv2.pt",
        device=0
    ):

        self.device = device

        self.model = YOLO(
            model_path,
            task="detect"
        )


        self.classes = [
            "sign board",
            "road traffic sign",
            "highway sign mounted on pole",
            "vertical sign pole"
        ]


        self.model.set_classes(
            self.classes
        )

    def box_iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter)

    def detect(
        self,
        image,
        min_conf=0.5
    ):

        results = self.model.predict(
            image,
            conf=min_conf,
            device=self.device,
            verbose=False
        )


        result = results[0]


        detections=[]


        for box in result.boxes:


            cls_id=int(
                box.cls[0]
            )


            conf=float(
                box.conf[0]
            )


            x1,y1,x2,y2 = map(
                int,
                box.xyxy[0]
            )


            detections.append(
                {
                    "class":
                        self.model.names[cls_id],

                    "confidence":
                        conf,

                    "bbox":
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                }
            )

        clean = []

        for det in detections:

            keep = True

            for old in clean:

                if self.box_iou(
                        det["bbox"],
                        old["bbox"]
                ) > 0.8:
                    keep = False

                    break

            if keep:
                clean.append(det)

        return detections