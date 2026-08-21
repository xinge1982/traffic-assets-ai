#!/usr/bin/env python3
"""
Generate true-depth calibration training data for traffic signs.

Each source row represents one sign detection. Images are downloaded once,
SAM2 and Depth Anything V3 are run with the same preprocessing used by
camera_sign_measure/main.py, and the resulting raw depth features are joined
with the sign center transformed into camera coordinates.

Camera coordinate convention:
    +X = camera right
    +Y = camera down
    +Z = camera forward
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urljoin

import cv2
import numpy as np
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CAMERA_MEASURE_DIR = REPO_ROOT / "camera_sign_measure"

if str(CAMERA_MEASURE_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_MEASURE_DIR))

from camera_geometry import median_depth_inside_mask  # noqa: E402
from depth_anything import DepthAnythingV3  # noqa: E402
from geometry import mask_to_contour_quad  # noqa: E402
from main import order_quad_points  # noqa: E402
from sam2_segmentor import SAM2Segmentor  # noqa: E402


DEFAULT_INPUT_CSV = SCRIPT_DIR / "input" / "camera_sign.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

POINT_Z_PATTERN = re.compile(
    r"^\s*POINT\s+Z?\s*\(\s*"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)"
    r"(?:\s+([-+0-9.eE]+))?\s*\)\s*$",
    re.IGNORECASE,
)

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download source images and generate DA3 true-depth "
            "training data for traffic signs."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Source CSV. Default: sign_true_depth/input/camera_sign.csv",
    )
    parser.add_argument(
        "--image-url-prefix",
        required=True,
        help=(
            "HTTP/HTTPS prefix prepended to each CSV img value, "
            "for example https://server.example/photos/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: sign_true_depth/output",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="sam2/checkpoints/sam2.1_hiera_small.pt",
        help="SAM2 checkpoint path relative to repository root.",
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config used by camera_sign_measure.",
    )
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
        help="Camera yaw offset from camera_heading in degrees.",
    )
    parser.add_argument(
        "--pitch-offset",
        type=float,
        default=0.0,
        help="Camera pitch in degrees; positive points upward.",
    )
    parser.add_argument(
        "--roll-offset",
        type=float,
        default=0.0,
        help="Camera roll offset in degrees.",
    )
    parser.add_argument(
        "--lever-forward",
        type=float,
        default=0.0,
        help="GNSS antenna to camera offset forward in meters.",
    )
    parser.add_argument(
        "--lever-right",
        type=float,
        default=0.0,
        help="GNSS antenna to camera offset right in meters.",
    )
    parser.add_argument(
        "--lever-up",
        type=float,
        default=0.0,
        help="GNSS antenna to camera offset upward in meters.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Image HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Download images again even when cached files exist.",
    )
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
        help="Rewrite cached depth and mask artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV rows to process.",
    )
    return parser.parse_args()


def parse_point_z(wkt):
    match = POINT_Z_PATTERN.match(str(wkt))
    if not match:
        raise ValueError(f"Invalid POINT Z WKT: {wkt}")

    longitude = float(match.group(1))
    latitude = float(match.group(2))
    altitude = float(match.group(3) or 0.0)
    return longitude, latitude, altitude


def geodetic_to_ecef(longitude, latitude, altitude):
    lon = math.radians(longitude)
    lat = math.radians(latitude)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    prime_vertical = WGS84_A / math.sqrt(
        1.0 - WGS84_E2 * sin_lat * sin_lat
    )

    x = (prime_vertical + altitude) * cos_lat * cos_lon
    y = (prime_vertical + altitude) * cos_lat * sin_lon
    z = (
        prime_vertical * (1.0 - WGS84_E2) + altitude
    ) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def ecef_delta_to_enu(delta_ecef, origin_longitude, origin_latitude):
    lon = math.radians(origin_longitude)
    lat = math.radians(origin_latitude)

    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    rotation = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [
                -sin_lat * cos_lon,
                -sin_lat * sin_lon,
                cos_lat,
            ],
            [
                cos_lat * cos_lon,
                cos_lat * sin_lon,
                sin_lat,
            ],
        ],
        dtype=np.float64,
    )

    return rotation @ delta_ecef


def camera_axes_enu(heading, pitch, roll):
    heading_rad = math.radians(heading)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    sin_heading = math.sin(heading_rad)
    cos_heading = math.cos(heading_rad)
    sin_pitch = math.sin(pitch_rad)
    cos_pitch = math.cos(pitch_rad)

    forward = np.array(
        [
            sin_heading * cos_pitch,
            cos_heading * cos_pitch,
            sin_pitch,
        ],
        dtype=np.float64,
    )

    right_zero_roll = np.array(
        [
            cos_heading,
            -sin_heading,
            0.0,
        ],
        dtype=np.float64,
    )

    down_zero_roll = np.cross(
        forward,
        right_zero_roll,
    )
    down_zero_roll /= np.linalg.norm(down_zero_roll)

    right = (
        right_zero_roll * math.cos(roll_rad)
        + down_zero_roll * math.sin(roll_rad)
    )
    down = (
        -right_zero_roll * math.sin(roll_rad)
        + down_zero_roll * math.cos(roll_rad)
    )

    return right, down, forward


def world_points_to_camera(
        photo_wkt,
        sign_wkt,
        camera_heading,
        yaw_offset,
        pitch_offset,
        roll_offset,
        lever_forward,
        lever_right,
        lever_up,
):
    photo_lon, photo_lat, photo_alt = parse_point_z(photo_wkt)
    sign_lon, sign_lat, sign_alt = parse_point_z(sign_wkt)

    photo_ecef = geodetic_to_ecef(
        photo_lon,
        photo_lat,
        photo_alt,
    )
    sign_ecef = geodetic_to_ecef(
        sign_lon,
        sign_lat,
        sign_alt,
    )

    sign_delta_enu = ecef_delta_to_enu(
        sign_ecef - photo_ecef,
        photo_lon,
        photo_lat,
    )

    right, down, forward = camera_axes_enu(
        float(camera_heading) + yaw_offset,
        pitch_offset,
        roll_offset,
    )

    camera_offset_enu = (
        forward * lever_forward
        + right * lever_right
        - down * lever_up
    )
    camera_to_sign_enu = (
        sign_delta_enu - camera_offset_enu
    )

    camera_x = float(np.dot(camera_to_sign_enu, right))
    camera_y = float(np.dot(camera_to_sign_enu, down))
    camera_z = float(np.dot(camera_to_sign_enu, forward))
    camera_range = float(np.linalg.norm(camera_to_sign_enu))

    return camera_x, camera_y, camera_z, camera_range


def validate_relative_image_path(image_value):
    normalized = str(image_value).replace("\\", "/").lstrip("/")
    relative = PurePosixPath(normalized)

    if not normalized or ".." in relative.parts:
        raise ValueError(f"Unsafe img path: {image_value}")

    return relative


def build_image_url(prefix, relative_path):
    safe_path = quote(
        relative_path.as_posix(),
        safe="/",
    )
    return urljoin(
        prefix.rstrip("/") + "/",
        safe_path,
    )


def download_image(
        session,
        image_url,
        destination,
        timeout,
        overwrite,
):
    if destination.exists() and not overwrite:
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    response = session.get(
        image_url,
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()

    with open(temporary, "wb") as output_file:
        for chunk in response.iter_content(
                chunk_size=1024 * 1024
        ):
            if chunk:
                output_file.write(chunk)

    temporary.replace(destination)


def depth_region_statistics(depth, mask, erosion_size=7):
    mask = np.asarray(mask).squeeze().astype(np.uint8)

    if mask.shape != depth.shape:
        mask = cv2.resize(
            mask,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    kernel = np.ones(
        (erosion_size, erosion_size),
        dtype=np.uint8,
    )
    inner_mask = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    valid = (
        (inner_mask > 0)
        & np.isfinite(depth)
        & (depth > 0)
    )

    if not np.any(valid):
        valid = (
            (mask > 0)
            & np.isfinite(depth)
            & (depth > 0)
        )

    if not np.any(valid):
        raise ValueError(
            "No valid DA3 depth inside sign mask"
        )

    values = depth[valid].astype(np.float64)
    raw_median = median_depth_inside_mask(
        depth,
        mask,
        erosion_size=erosion_size,
    )

    return {
        "raw_depth_median": float(raw_median),
        "raw_depth_mean": float(np.mean(values)),
        "raw_depth_std": float(np.std(values)),
        "raw_depth_min": float(np.min(values)),
        "raw_depth_max": float(np.max(values)),
        "raw_depth_p10": float(np.percentile(values, 10)),
        "raw_depth_p90": float(np.percentile(values, 90)),
        "depth_valid_count": int(values.size),
        "depth_valid_ratio": float(
            values.size / max(1, np.count_nonzero(mask))
        ),
    }


def safe_artifact_name(sample_id):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(sample_id),
    )


def process_detection(
        row,
        image,
        resized_depth,
        depth,
        depth_k,
        mask,
        output_dir,
        overwrite_artifacts,
        pose_options,
):
    image_height, image_width = image.shape[:2]
    depth_height, depth_width = depth.shape[:2]

    corners = mask_to_contour_quad(mask)
    if corners is None or len(corners) != 4:
        raise ValueError(
            "Failed to obtain four sign corners from SAM2 mask"
        )

    corners = order_quad_points(corners)
    stats = depth_region_statistics(
        resized_depth,
        mask,
    )

    camera_x, camera_y, camera_z, camera_range = (
        world_points_to_camera(
            row["photo_wkt"],
            row["sign_wkt"],
            row["camera_heading"],
            **pose_options,
        )
    )

    raw_depth = stats["raw_depth_median"]
    target_scale = (
        camera_z / raw_depth
        if raw_depth > 0
        else float("nan")
    )

    sample_name = safe_artifact_name(
        row.get("id", "sample")
    )
    mask_relative = Path("masks") / f"{sample_name}.png"
    mask_path = output_dir / mask_relative

    if overwrite_artifacts or not mask_path.exists():
        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        cv2.imwrite(
            str(mask_path),
            mask.astype(np.uint8) * 255,
        )

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = float(image_width * image_height)
    mask_area = int(np.count_nonzero(mask))

    result = dict(row)
    result.update(stats)
    result.update(
        {
            "sign_group_id": row.get("feature_id", ""),
            "image_width": image_width,
            "image_height": image_height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "depth_fx": float(depth_k[0, 0]),
            "depth_fy": float(depth_k[1, 1]),
            "depth_cx": float(depth_k[0, 2]),
            "depth_cy": float(depth_k[1, 2]),
            "corners_json": json.dumps(
                corners.astype(float).tolist(),
                separators=(",", ":"),
            ),
            "mask_path": mask_relative.as_posix(),
            "mask_area": mask_area,
            "mask_area_ratio": mask_area / image_area,
            "bbox_area_ratio": bbox_area / image_area,
            "center_u_normalized": (
                float(row["u"]) / image_width
            ),
            "center_v_normalized": (
                float(row["v"]) / image_height
            ),
            "true_camera_x": camera_x,
            "true_camera_y": camera_y,
            "true_camera_z": camera_z,
            "true_camera_range": camera_range,
            "target_depth_scale": target_scale,
            "label_source": "photo_wkt_sign_wkt_heading",
            "pose_yaw_offset": pose_options["yaw_offset"],
            "pose_pitch_offset": pose_options["pitch_offset"],
            "pose_roll_offset": pose_options["roll_offset"],
            "lever_forward": pose_options["lever_forward"],
            "lever_right": pose_options["lever_right"],
            "lever_up": pose_options["lever_up"],
            "status": (
                "ok"
                if camera_z > 0
                else "invalid_behind_camera"
            ),
            "error": "",
        }
    )

    return result


def output_fieldnames(source_fieldnames):
    generated = [
        "sign_group_id",
        "image_url",
        "local_image_path",
        "depth_path",
        "mask_path",
        "image_width",
        "image_height",
        "depth_width",
        "depth_height",
        "depth_fx",
        "depth_fy",
        "depth_cx",
        "depth_cy",
        "corners_json",
        "mask_area",
        "mask_area_ratio",
        "bbox_area_ratio",
        "center_u_normalized",
        "center_v_normalized",
        "raw_depth_median",
        "raw_depth_mean",
        "raw_depth_std",
        "raw_depth_min",
        "raw_depth_max",
        "raw_depth_p10",
        "raw_depth_p90",
        "depth_valid_count",
        "depth_valid_ratio",
        "true_camera_x",
        "true_camera_y",
        "true_camera_z",
        "true_camera_range",
        "target_depth_scale",
        "label_source",
        "pose_yaw_offset",
        "pose_pitch_offset",
        "pose_roll_offset",
        "lever_forward",
        "lever_right",
        "lever_up",
        "status",
        "error",
    ]

    fields = list(source_fieldnames)
    for field in generated:
        if field not in fields:
            fields.append(field)
    return fields


def read_source_rows(input_csv, limit):
    with open(
            input_csv,
            "r",
            encoding="utf-8-sig",
            newline=""
    ) as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        required = {
            "id",
            "img",
            "x1",
            "y1",
            "x2",
            "y2",
            "camera_heading",
            "u",
            "v",
            "photo_wkt",
            "sign_wkt",
        }
        missing = sorted(
            required - set(reader.fieldnames)
        )
        if missing:
            raise ValueError(
                "Missing input CSV fields: "
                + ", ".join(missing)
            )

        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

        return reader.fieldnames, rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
            path,
            "w",
            encoding="utf-8-sig",
            newline=""
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_cache_dir = output_dir / "images"
    depth_dir = output_dir / "depth"

    source_fieldnames, source_rows = read_source_rows(
        input_csv,
        args.limit,
    )

    rows_by_image = defaultdict(list)
    for row in source_rows:
        rows_by_image[row["img"]].append(row)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_options = {
        "yaw_offset": args.yaw_offset,
        "pitch_offset": args.pitch_offset,
        "roll_offset": args.roll_offset,
        "lever_forward": args.lever_forward,
        "lever_right": args.lever_right,
        "lever_up": args.lever_up,
    }

    original_working_directory = Path.cwd()
    os.chdir(REPO_ROOT)

    print("Loading SAM2...")
    sam2 = SAM2Segmentor(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
    )

    print("Loading Depth Anything V3...")
    depth_model = DepthAnythingV3()

    successful_rows = []
    failed_rows = []
    session = requests.Session()

    try:
        total_images = len(rows_by_image)

        for image_index, (image_value, rows) in enumerate(
                rows_by_image.items(),
                start=1
        ):
            print(
                f"[{image_index}/{total_images}] "
                f"Processing {image_value} "
                f"({len(rows)} signs)"
            )

            try:
                relative_image = validate_relative_image_path(
                    image_value
                )
                image_url = build_image_url(
                    args.image_url_prefix,
                    relative_image,
                )
                local_image_path = image_cache_dir.joinpath(
                    *relative_image.parts
                )

                download_image(
                    session,
                    image_url,
                    local_image_path,
                    args.request_timeout,
                    args.overwrite_images,
                )

                image = cv2.imread(
                    str(local_image_path),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise ValueError(
                        f"Unable to decode image: {local_image_path}"
                    )

                image_height, image_width = image.shape[:2]
                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                sam2.set_image(image_rgb)
                depth_prediction = depth_model.predict(
                    image_rgb
                )
                depth_k = (
                    depth_prediction.intrinsics[0].copy()
                )
                depth = depth_prediction.depth[0]
                resized_depth = cv2.resize(
                    depth,
                    (image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                depth_relative = (
                    Path("depth")
                    / relative_image.with_suffix(".npy")
                )
                depth_path = output_dir / depth_relative

                if (
                    args.overwrite_artifacts
                    or not depth_path.exists()
                ):
                    depth_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    np.save(
                        depth_path,
                        depth.astype(np.float32),
                    )

                for row in rows:
                    try:
                        bbox = np.array(
                            [
                                float(row["x1"]),
                                float(row["y1"]),
                                float(row["x2"]),
                                float(row["y2"]),
                            ],
                            dtype=np.float32,
                        )

                        bbox[0::2] = np.clip(
                            bbox[0::2],
                            0,
                            image_width - 1,
                        )
                        bbox[1::2] = np.clip(
                            bbox[1::2],
                            0,
                            image_height - 1,
                        )

                        if (
                            bbox[2] <= bbox[0]
                            or bbox[3] <= bbox[1]
                        ):
                            raise ValueError(
                                f"Invalid bbox after clipping: {bbox}"
                            )

                        segment_result = sam2.segment_box(
                            bbox.tolist()
                        )
                        mask = segment_result["mask"]

                        if hasattr(mask, "cpu"):
                            mask = mask.cpu().numpy()

                        mask = (
                            np.squeeze(mask)
                            .astype(np.uint8)
                        )

                        result = process_detection(
                            row,
                            image,
                            resized_depth,
                            depth,
                            depth_k,
                            mask,
                            output_dir,
                            args.overwrite_artifacts,
                            pose_options,
                        )
                        result["image_url"] = image_url
                        result["local_image_path"] = (
                            local_image_path
                            .relative_to(output_dir)
                            .as_posix()
                        )
                        result["depth_path"] = (
                            depth_relative.as_posix()
                        )

                        if result["status"] == "ok":
                            successful_rows.append(result)
                        else:
                            failed_rows.append(result)

                    except Exception as error:
                        failed = dict(row)
                        failed.update(
                            {
                                "image_url": image_url,
                                "local_image_path": (
                                    local_image_path
                                    .relative_to(output_dir)
                                    .as_posix()
                                ),
                                "depth_path": (
                                    depth_relative.as_posix()
                                ),
                                "status": "error",
                                "error": str(error),
                            }
                        )
                        failed_rows.append(failed)
                        print(
                            f"  Failed {row.get('id')}: {error}"
                        )

            except Exception as image_error:
                print(
                    f"  Image failed: {image_error}"
                )
                for row in rows:
                    failed = dict(row)
                    failed.update(
                        {
                            "status": "error",
                            "error": str(image_error),
                        }
                    )
                    failed_rows.append(failed)

            fieldnames = output_fieldnames(
                source_fieldnames
            )
            write_csv(
                output_dir / "training_data.csv",
                fieldnames,
                successful_rows,
            )
            write_csv(
                output_dir / "failed_samples.csv",
                fieldnames,
                failed_rows,
            )

    finally:
        session.close()
        os.chdir(original_working_directory)

    print(
        f"Generated {len(successful_rows)} training samples."
    )
    print(
        f"Failed or invalid samples: {len(failed_rows)}."
    )
    print(
        "Training CSV:",
        output_dir / "training_data.csv",
    )
    print(
        "Failure CSV:",
        output_dir / "failed_samples.csv",
    )


if __name__ == "__main__":
    main()
