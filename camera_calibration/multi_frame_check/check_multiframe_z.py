#!/usr/bin/env python3
"""Inspect multi-frame consistency of GIS-derived camera coordinates.

This script is deliberately a diagnostic step, not a pose optimizer.  It
converts each manually verified photo/sign association into camera coordinates
(+X right, +Y down, +Z forward), then checks whether repeated observations of
the same feature change consistently over time.

It does not need images, SAM, DA3, or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
POINT_Z_PATTERN = re.compile(
    r"^\s*POINT\s+Z?\s*\(\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    r"(?:\s+([-+0-9.eE]+))?\s*\)\s*$",
    re.IGNORECASE,
)
FRAME_PATTERN = re.compile(r"/(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether GIS-derived camera Z values are temporally "
            "consistent across repeated observations of each feature."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--output-dir",
        default="multiframe_z_output",
        help="Directory for diagnostic CSV files.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="Candidate camera yaw correction, clockwise degrees.",
    )
    parser.add_argument(
        "--pitch-offset-deg",
        type=float,
        default=0.0,
        help="Camera pitch correction; positive points upward.",
    )
    parser.add_argument(
        "--roll-offset-deg",
        type=float,
        default=0.0,
        help="Camera roll correction.",
    )
    parser.add_argument("--lever-forward-m", type=float, default=0.0)
    parser.add_argument("--lever-right-m", type=float, default=0.0)
    parser.add_argument("--lever-up-m", type=float, default=0.0)
    parser.add_argument(
        "--principal-point-x-px",
        type=float,
        default=1920.0,
        help="Image centre used only for left/right consistency checks.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=2,
        help="Larger gaps are flagged instead of compared as adjacent frames.",
    )
    parser.add_argument(
        "--max-z-residual-m",
        type=float,
        default=3.0,
        help="Absolute Z-motion residual beyond which a pair is suspicious.",
    )
    return parser.parse_args()


def parse_point_z(wkt: str) -> tuple[float, float, float]:
    match = POINT_Z_PATTERN.match(str(wkt))
    if not match:
        raise ValueError(f"Invalid POINT Z WKT: {wkt}")
    return tuple(float(match.group(index) or 0.0) for index in (1, 2, 3))


def geodetic_to_ecef(longitude: float, latitude: float, altitude: float) -> np.ndarray:
    longitude_rad = math.radians(longitude)
    latitude_rad = math.radians(latitude)
    sin_lat = math.sin(latitude_rad)
    cos_lat = math.cos(latitude_rad)
    sin_lon = math.sin(longitude_rad)
    cos_lon = math.cos(longitude_rad)
    prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (prime_vertical + altitude) * cos_lat * cos_lon,
            (prime_vertical + altitude) * cos_lat * sin_lon,
            (prime_vertical * (1.0 - WGS84_E2) + altitude) * sin_lat,
        ],
        dtype=np.float64,
    )


def ecef_delta_to_enu(delta_ecef: np.ndarray, longitude: float, latitude: float) -> np.ndarray:
    longitude_rad = math.radians(longitude)
    latitude_rad = math.radians(latitude)
    sin_lon = math.sin(longitude_rad)
    cos_lon = math.cos(longitude_rad)
    sin_lat = math.sin(latitude_rad)
    cos_lat = math.cos(latitude_rad)
    rotation = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return rotation @ delta_ecef


def camera_axes_enu(heading_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    forward = np.array(
        [math.sin(heading) * math.cos(pitch), math.cos(heading) * math.cos(pitch), math.sin(pitch)],
        dtype=np.float64,
    )
    right_zero_roll = np.array([math.cos(heading), -math.sin(heading), 0.0], dtype=np.float64)
    down_zero_roll = np.cross(forward, right_zero_roll)
    down_zero_roll /= np.linalg.norm(down_zero_roll)
    right = right_zero_roll * math.cos(roll) + down_zero_roll * math.sin(roll)
    down = -right_zero_roll * math.sin(roll) + down_zero_roll * math.cos(roll)
    return right, down, forward


def camera_coordinates(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    photo_lon, photo_lat, photo_alt = parse_point_z(row["photo_wkt"])
    sign_lon, sign_lat, sign_alt = parse_point_z(row["sign_wkt"])
    photo_ecef = geodetic_to_ecef(photo_lon, photo_lat, photo_alt)
    sign_ecef = geodetic_to_ecef(sign_lon, sign_lat, sign_alt)
    sign_delta_enu = ecef_delta_to_enu(sign_ecef - photo_ecef, photo_lon, photo_lat)
    heading = float(row["camera_heading"]) + args.yaw_offset_deg
    right, down, forward = camera_axes_enu(heading, args.pitch_offset_deg, args.roll_offset_deg)
    camera_offset_enu = (
        forward * args.lever_forward_m
        + right * args.lever_right_m
        - down * args.lever_up_m
    )
    camera_to_sign_enu = sign_delta_enu - camera_offset_enu
    return {
        "photo_lon": photo_lon,
        "photo_lat": photo_lat,
        "photo_alt_m": photo_alt,
        "sign_delta_east_m": float(sign_delta_enu[0]),
        "sign_delta_north_m": float(sign_delta_enu[1]),
        "sign_delta_up_m": float(sign_delta_enu[2]),
        "camera_x_raw_m": float(np.dot(camera_to_sign_enu, right)),
        "camera_y_raw_m": float(np.dot(camera_to_sign_enu, down)),
        "camera_z_raw_m": float(np.dot(camera_to_sign_enu, forward)),
        "camera_range_raw_m": float(np.linalg.norm(camera_to_sign_enu)),
        "camera_forward_e": float(forward[0]),
        "camera_forward_n": float(forward[1]),
        "camera_forward_u": float(forward[2]),
    }


def sequence_and_frame(row: dict[str, str]) -> tuple[str, int]:
    image = row.get("img") or row.get("image") or ""
    sequence = image.split("/")[0] if "/" in image else ""
    match = FRAME_PATTERN.search(image)
    if not match:
        raise ValueError(f"Cannot extract frame number from image: {image}")
    return sequence, int(match.group(1))


def require_columns(fieldnames: list[str] | None) -> None:
    required = {"img", "feature_id", "camera_heading", "u", "v", "photo_wkt", "sign_wkt", "x1", "y1", "x2", "y2"}
    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError("Missing required CSV fields: " + ", ".join(missing))


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def make_pair_diagnostics(previous: dict[str, Any], current: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    frame_gap = current["frame_number"] - previous["frame_number"]
    prev_ecef = geodetic_to_ecef(previous["photo_lon"], previous["photo_lat"], previous["photo_alt_m"])
    curr_ecef = geodetic_to_ecef(current["photo_lon"], current["photo_lat"], current["photo_alt_m"])
    displacement_enu = ecef_delta_to_enu(curr_ecef - prev_ecef, previous["photo_lon"], previous["photo_lat"])
    previous_forward = np.array(
        [previous["camera_forward_e"], previous["camera_forward_n"], previous["camera_forward_u"]],
        dtype=np.float64,
    )
    motion_forward = float(np.dot(displacement_enu, previous_forward))
    delta_z = current["camera_z_raw_m"] - previous["camera_z_raw_m"]
    expected_delta_z = -motion_forward
    residual = delta_z - expected_delta_z
    delta_u = as_float(current, "u") - as_float(previous, "u")
    delta_area = current["bbox_area_px"] - previous["bbox_area_px"]
    status = "ok"
    if frame_gap <= 0:
        status = "invalid_frame_order"
    elif frame_gap > args.max_frame_gap:
        status = "large_frame_gap"
    elif previous["camera_z_raw_m"] <= 0 or current["camera_z_raw_m"] <= 0:
        status = "behind_camera"
    elif abs(residual) > args.max_z_residual_m:
        status = "z_motion_residual_large"
    return {
        "previous_frame_number": previous["frame_number"],
        "frame_gap": frame_gap,
        "motion_forward_m": motion_forward,
        "motion_lateral_m": float(np.linalg.norm(displacement_enu - motion_forward * previous_forward)),
        "delta_z_m": delta_z,
        "expected_delta_z_m": expected_delta_z,
        "z_temporal_residual_m": residual,
        "delta_u_px": delta_u,
        "delta_bbox_area_px": delta_area,
        "pair_status": status,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        require_columns(reader.fieldnames)
        source_rows = list(reader)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_row in source_rows:
        try:
            sequence_id, frame_number = sequence_and_frame(source_row)
            result: dict[str, Any] = dict(source_row)
            result.update(camera_coordinates(source_row, args))
            result["sequence_id"] = sequence_id
            result["frame_number"] = frame_number
            result["bbox_width_px"] = as_float(source_row, "x2") - as_float(source_row, "x1")
            result["bbox_height_px"] = as_float(source_row, "y2") - as_float(source_row, "y1")
            result["bbox_area_px"] = result["bbox_width_px"] * result["bbox_height_px"]
            result["camera_front_check"] = "ok" if result["camera_z_raw_m"] > 0 else "behind_camera"
            image_side = as_float(source_row, "u") - args.principal_point_x_px
            result["image_side"] = "right" if image_side > 0 else "left" if image_side < 0 else "center"
            result["geometry_side"] = "right" if result["camera_x_raw_m"] > 0 else "left" if result["camera_x_raw_m"] < 0 else "center"
            result["side_consistency"] = "ok" if image_side * result["camera_x_raw_m"] >= 0 else "mismatch"
            result.update({
                "previous_frame_number": "",
                "frame_gap": "",
                "motion_forward_m": "",
                "motion_lateral_m": "",
                "delta_z_m": "",
                "expected_delta_z_m": "",
                "z_temporal_residual_m": "",
                "delta_u_px": "",
                "delta_bbox_area_px": "",
                "pair_status": "single_frame",
            })
            rows.append(result)
            grouped[(sequence_id, source_row["feature_id"])].append(result)
        except Exception as error:
            failed = dict(source_row)
            failed["status"] = "error"
            failed["error"] = str(error)
            failures.append(failed)

    group_summary: list[dict[str, Any]] = []
    for (sequence_id, feature_id), observations in grouped.items():
        observations.sort(key=lambda item: item["frame_number"])
        for index, current in enumerate(observations):
            if index == 0:
                continue
            current.update(make_pair_diagnostics(observations[index - 1], current, args))

        pair_rows = observations[1:]
        pair_statuses = [item["pair_status"] for item in pair_rows]
        group_status = "single_frame" if len(observations) == 1 else "ok"
        if any(item["camera_front_check"] != "ok" for item in observations):
            group_status = "behind_camera"
        elif any(status != "ok" for status in pair_statuses):
            group_status = "needs_review"
        group_summary.append(
            {
                "sequence_id": sequence_id,
                "feature_id": feature_id,
                "observation_count": len(observations),
                "first_frame_number": observations[0]["frame_number"],
                "last_frame_number": observations[-1]["frame_number"],
                "z_first_m": observations[0]["camera_z_raw_m"],
                "z_last_m": observations[-1]["camera_z_raw_m"],
                "z_change_m": observations[-1]["camera_z_raw_m"] - observations[0]["camera_z_raw_m"],
                "max_abs_z_temporal_residual_m": max((abs(float(item["z_temporal_residual_m"])) for item in pair_rows if item["z_temporal_residual_m"] != ""), default=""),
                "group_status": group_status,
            }
        )

    rows.sort(key=lambda item: (item["sequence_id"], item["feature_id"], item["frame_number"]))
    group_summary.sort(key=lambda item: (item["sequence_id"], item["first_frame_number"], item["feature_id"]))
    write_csv(output_dir / "multiframe_z_rows.csv", rows)
    write_csv(output_dir / "multiframe_z_groups.csv", group_summary)
    write_csv(output_dir / "failed_rows.csv", failures)

    summary = {
        "input_csv": str(input_path),
        "input_rows": len(source_rows),
        "successful_rows": len(rows),
        "failed_rows": len(failures),
        "feature_groups": len(group_summary),
        "multi_frame_groups": sum(item["observation_count"] >= 2 for item in group_summary),
        "behind_camera_rows": sum(item["camera_front_check"] == "behind_camera" for item in rows),
        "groups_needing_review": sum(item["group_status"] == "needs_review" for item in group_summary),
        "parameters": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Rows:", output_dir / "multiframe_z_rows.csv")
    print("Groups:", output_dir / "multiframe_z_groups.csv")


if __name__ == "__main__":
    main()
