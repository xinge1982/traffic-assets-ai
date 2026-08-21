import base64
import json
import os
import uuid
from PIL import Image

import cv2
import numpy as np

from yolo_detector import YOLODetector
from sam2_segmentor import SAM2Segmentor
from geometry import mask_to_contour_quad
from depth_anything import DepthAnythingV3
from camera_geometry import calculate_sign_size_3d

output_dir = "output"
output_json = os.path.join(output_dir, "sign_models.json")
camera_calibration_file = os.getenv(
    "CAMERA_CALIBRATION_FILE",
    "input/camera_calibration_jinji.json"
)

# -----------------------
# Global Models
# -----------------------

_detector = None
_sam2 = None
_depth_model = None

def initialize_pipeline():
    global _detector
    global _sam2
    global _depth_model

    if _detector is None:
        print("Loading YOLO...")
        _detector = YOLODetector(
            "models/yolov8l-worldv2.pt"
        )

    if _sam2 is None:
        print("Loading SAM2...")
        _sam2 = SAM2Segmentor(
            checkpoint="sam2/checkpoints/sam2.1_hiera_small.pt",
            config="configs/sam2.1/sam2.1_hiera_s.yaml"
        )

    if _depth_model is None:
        print("Loading Depth Anything...")
        _depth_model = DepthAnythingV3()

    print("Pipeline initialized.")

def image_file_to_base64(image_path):
    """
    Read image file and convert to base64 data URI.

    Args:
        image_path: image filename

    Returns:
        str:
        data:image/jpeg;base64,...
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # detect image type
    ext = image_path.lower().split(".")[-1]

    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    return f"data:{mime};base64,{encoded}"

def decode_base64_image(image_base64):

    # remove:
    # data:image/jpeg;base64,
    # data:image/png;base64,

    if "," in image_base64:
        image_base64 = image_base64.split(",",1)[1]


    image_bytes = base64.b64decode(
        image_base64
    )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Invalid image base64"
        )

    return image



def load_camera_calibration(
        filename,
        image_width,
        image_height
):
    """
    Load Brown-Conrady camera calibration parameters from JSON.

    Supported formats:
        camera_matrix + distortion_coefficients
    or:
        fx, fy, cx, cy + k1, k2, p1, p2, k3

    image_width/image_height in the file describe the resolution used
    during calibration. Intrinsics are scaled to the current image.
    """
    with open(filename, "r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    if "camera_matrix" in calibration:
        camera_matrix = np.asarray(
            calibration["camera_matrix"],
            dtype=np.float64
        ).reshape(3, 3)
    else:
        required_intrinsics = (
            "fx",
            "fy",
            "cx",
            "cy"
        )
        missing = [
            key
            for key in required_intrinsics
            if key not in calibration
        ]
        if missing:
            raise ValueError(
                "Missing camera calibration fields: "
                + ", ".join(missing)
            )

        camera_matrix = np.array(
            [
                [calibration["fx"], 0.0, calibration["cx"]],
                [0.0, calibration["fy"], calibration["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64
        )

    if "distortion_coefficients" in calibration:
        distortion_coefficients = np.asarray(
            calibration["distortion_coefficients"],
            dtype=np.float64
        ).reshape(-1)
    else:
        distortion_coefficients = np.array(
            [
                calibration.get("k1", 0.0),
                calibration.get("k2", 0.0),
                calibration.get("p1", 0.0),
                calibration.get("p2", 0.0),
                calibration.get("k3", 0.0),
            ],
            dtype=np.float64
        )

    calibration_width = float(
        calibration.get("image_width", image_width)
    )
    calibration_height = float(
        calibration.get("image_height", image_height)
    )

    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError(
            "Calibration image_width and image_height must be positive."
        )

    scale_x = image_width / calibration_width
    scale_y = image_height / calibration_height

    camera_matrix = camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[1, 2] *= scale_y

    return camera_matrix, distortion_coefficients


def save_depth(depth, filename):
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))

    if depth_max <= depth_min:
        depth_img = np.zeros(depth.shape, dtype=np.uint8)
    else:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_img = (depth_norm * 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    cv2.imwrite(filename, depth_color)


def order_quad_points(points):
    """
    Convert arbitrary four corner points to:
        top-left, top-right, bottom-right, bottom-left
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sums)]   # top-left
    ordered[2] = points[np.argmax(point_sums)]   # bottom-right
    ordered[1] = points[np.argmin(point_diffs)]  # top-right
    ordered[3] = points[np.argmax(point_diffs)]  # bottom-left

    return ordered


def perspective_crop_to_base64(image, corners):
    """
    Perspective-transform the four sign corners into a rectangular PNG
    and return:
        data:image/png;base64,...
    """
    src = order_quad_points(corners)

    top_width = np.linalg.norm(src[1] - src[0])
    bottom_width = np.linalg.norm(src[2] - src[3])
    left_height = np.linalg.norm(src[3] - src[0])
    right_height = np.linalg.norm(src[2] - src[1])

    target_width = max(1, int(round(max(top_width, bottom_width))))
    target_height = max(1, int(round(max(left_height, right_height))))

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        image,
        perspective_matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    success, encoded = cv2.imencode(".png", warped)
    if not success:
        raise RuntimeError("Failed to encode perspective-cropped sign image.")

    encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_text}", warped


def camera_point_to_blender(point):
    """
    Current camera coordinates from pixel_to_xyz():
        +X = image/camera right
        +Y = image/camera down
        +Z = camera forward

    Blender right-handed coordinates used here:
        +X = right
        +Y = forward
        +Z = up

    Therefore:
        blender_x = camera_x
        blender_y = camera_z
        blender_z = -camera_y
    """
    camera_x, camera_y, camera_z = map(float, point)

    return np.array(
        [
            camera_x,
            camera_z,
            -camera_y,
        ],
        dtype=np.float64,
    )


def calculate_translation_blender(points):
    """
    Use the 3D center of the sign plate as its model translation.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center_camera = points.mean(axis=0)
    center_blender = camera_point_to_blender(center_camera)

    return {
        "x": round(float(center_blender[0]), 6),
        "y": round(float(center_blender[1]), 6),
        "z": round(float(center_blender[2]), 6),
    }


def create_model_json(sign_result, image_base64):
    return {
        "id": str(uuid.uuid4()),
        "modelerType": "image_sign",
        "role": "hdSign",
        "transform": {
            "translation": calculate_translation_blender(
                sign_result["points"]
            ),
            "rotation": {
                "x": 0,
                "y": 0,
                "z": 0,
                "w": 1,
            },
            "scale": {
                "x": 1,
                "y": 1,
                "z": 1,
            },
        },
        "content": {
            "width": round(float(sign_result["width"]), 6),
            "height": round(float(sign_result["height"]), 6),
            "depth": 0.02,
            "base64": image_base64,
        },
    }

def normalize_model_translation(models):
    """
    Normalize blender translation coordinates.
    x:
        subtract median x
    y:
        subtract minimum y
    z:
        subtract minimum z
    """
    if not models:
        return

    # -----------------------
    # X use median
    # -----------------------
    x_values = [
        model["transform"]["translation"]["x"]
        for model in models
    ]
    center_x = float(
        np.median(x_values)
    )

    # -----------------------
    # Y/Z use minimum
    # -----------------------
    y_values = [
        model["transform"]["translation"]["y"]
        for model in models
    ]
    z_values = [
        model["transform"]["translation"]["z"]
        for model in models
    ]

    min_y = min(y_values)
    min_z = min(z_values)

    # -----------------------
    # apply offset
    # -----------------------
    for model in models:
        translation = (
            model["transform"]["translation"]
        )

        translation["x"] = round(
            translation["x"] - center_x,
            6
        )
        translation["y"] = round(
            translation["y"] - min_y,
            6
        )
        translation["z"] = round(
            translation["z"] - min_z,
            6
        )

    return models

def run_sign_pipeline(
        image_base64,
        calibration_path=camera_calibration_file
):
    os.makedirs(output_dir, exist_ok=True)

    image = decode_base64_image(
        image_base64
    )
    if image is None:
        raise FileNotFoundError(f"Unable to decode image")

    image_height, image_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    camera_matrix, distortion_coefficients = load_camera_calibration(
        calibration_path,
        image_width,
        image_height
    )

    print("camera calibration:", calibration_path)
    print("camera matrix:", camera_matrix)
    print("distortion coefficients:", distortion_coefficients)

    detections = _detector.detect(
        image,
        min_conf=0.5,
    )

    print("detections:", detections)

    _sam2.set_image(image_rgb)

    # Run Depth Anything only once for the entire source image.
    depth_pred = _depth_model.predict(image_rgb)

    K = depth_pred.intrinsics[0].copy()
    depth = depth_pred.depth[0]
    depth_height, depth_width = depth.shape[:2]

    print("K:", K)
    print("depth shape:", depth.shape)

    save_depth(
        depth,
        os.path.join(output_dir, "depth.jpg"),
    )

    resized_depth = cv2.resize(
        depth,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )

    output_data = {
        "meta": {
            "schemaVersion": "3.0",
            "compositeType": "user_custom",
            "name": "",
            "axisSystem": "blender",
        },
        "models": [],
    }

    for index, detection in enumerate(detections):
        bbox = detection["bbox"]
        segment_result = _sam2.segment_box(bbox)
        mask = segment_result["mask"]

        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()

        mask = np.squeeze(mask).astype(np.uint8)

        # Convert to 1-bit image
        mask_1bit = Image.fromarray(mask * 255).convert("1")

        mask_1bit.save("mask.png", optimize=True)


        print(
            detection["class"],
            "mask area:",
            int(mask.sum()),
        )

        corners = mask_to_contour_quad(mask)
        if corners is None or len(corners) != 4:
            print(
                f"Skip detection {index}: failed to obtain four corners."
            )
            continue

        corners = order_quad_points(corners)

        print(
            "corners:",
            corners.tolist(),
        )

        sign_result = calculate_sign_size_3d(
            corners.astype(np.int32),
            resized_depth,
            mask,
            K,
            image_width,
            image_height,
            depth_width,
            depth_height,
            camera_matrix,
            distortion_coefficients,
        )

        print(
            "Sign:",
            sign_result,
        )

        if sign_result is None:
            continue

        if sign_result['width'] < 0.2 or sign_result['height'] < 0.2:
            continue

        image_base64, warped_sign = perspective_crop_to_base64(
            image,
            corners,
        )

        # Optional standalone image for checking the perspective result.
        cv2.imwrite(
            os.path.join(
                output_dir,
                f"sign_rectified_{index}.png",
            ),
            warped_sign,
        )

        model_json = create_model_json(
            sign_result,
            image_base64,
        )
        output_data["models"].append(model_json)

        # Save SAM2 overlay image.
        overlay = image.copy()
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = 255

        mask_pixels = mask > 0
        overlay[mask_pixels] = (
            0.5 * overlay[mask_pixels]
            + 0.5 * color_mask[mask_pixels]
        ).astype(np.uint8)

        cv2.polylines(
            overlay,
            [corners.astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=3,
        )

        cv2.imwrite(
            os.path.join(
                output_dir,
                f"sam2_overlay_{index}.jpg",
            ),
            overlay,
        )

    # -----------------------
    # Normalize translation
    # -----------------------
    output_data["models"] = normalize_model_translation(output_data["models"])

    return output_data


if __name__ == "__main__":
    initialize_pipeline()

    image_base64 = image_file_to_base64("input/0001603.jpeg")
    result = run_sign_pipeline(
        image_base64
    )

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(
            result,
            json_file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"Generated {len(result['models'])} sign models."
    )
    print(
        f"JSON saved to: {output_json}"
    )