from ultralytics import YOLO
import cv2
import os
from pathlib import Path


# Initialize YOLO-World model
model = YOLO(
    "yolov8l-worldv2.pt",
    task="detect"
)


# Define custom classes
classes = [
    "sign board",
    "road traffic sign",
    "highway sign mounted on pole",
    "vertical sign pole"
]

model.set_classes(classes)

# Stage 2:
# segmentation model
segmentor = YOLO(
    "yolo11l-seg.pt",
    task="segment"
)


# Folder settings
input_dir = Path("input")
output_dir = Path("output")

output_dir.mkdir(
    exist_ok=True
)


# Supported images
image_exts = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# Process every image
for img_path in input_dir.iterdir():

    if img_path.suffix.lower() not in image_exts:
        continue


    print("\nProcessing:", img_path)


    results = model.predict(
        str(img_path),
        conf=0.3,
        device=0,
        verbose=False
    )


    result = results[0]


    print(
        "Detected objects:",
        len(result.boxes)
    )


    # Print detections
    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        xyxy = box.xyxy[0].tolist()

        print(
            f"{model.names[cls_id]} "
            f"conf={conf:.3f} "
            f"box={xyxy}"
        )


    # Draw boxes
    rendered = result.plot(
        labels=True,
        conf=True
    )


    # Output filename
    output_file = output_dir / img_path.name


    cv2.imwrite(
        str(output_file),
        rendered
    )


    print(
        "Saved:",
        output_file
    )


print("\nAll images finished.")