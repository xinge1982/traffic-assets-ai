#!/usr/bin/env python3
"""Estimate fx, fy, cx, cy from camera/sign geodetic positions and image pixels.

Camera convention: x right, y down, z forward. Heading is degrees clockwise
from true north. Pitch is positive looking up; roll is positive clockwise when
looking forward. The lever arm is GNSS antenna -> camera centre in vehicle
Forward/Right/Up axes, in metres. The WKT order is longitude, latitude,
ellipsoidal/orthometric height; camera and sign heights must use the same datum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WKT_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_wkt_point_z(text: str) -> np.ndarray:
    values = [float(x) for x in WKT_NUMBER.findall(text)]
    if len(values) != 3:
        raise ValueError(f"expected POINT Z with 3 numbers, got {text!r}")
    return np.asarray(values, dtype=float)  # longitude, latitude, height


def utm_zone(longitude_deg: float) -> int:
    """Return the standard UTM longitudinal zone number (1..60)."""
    return min(60, max(1, int(math.floor((longitude_deg + 180.0) / 6.0)) + 1))


def automatic_utm_epsg(longitude_deg: float, latitude_deg: float) -> int:
    """Select WGS84 UTM EPSG from a representative longitude/latitude."""
    if not -80.0 <= latitude_deg <= 84.0:
        raise ValueError("automatic UTM requires latitude in [-80, 84] degrees")
    return (32600 if latitude_deg >= 0.0 else 32700) + utm_zone(longitude_deg)


def wgs84_to_utm(longitude_deg: float, latitude_deg: float, zone: int) -> tuple[float, float]:
    """Convert WGS84 lon/lat to UTM easting/northing without external GIS libs."""
    if not -80.0 <= latitude_deg <= 84.0:
        raise ValueError("UTM requires latitude in [-80, 84] degrees")
    a, f, k0 = 6378137.0, 1.0 / 298.257223563, 0.9996
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    sl, cl, tl = math.sin(lat), math.cos(lat), math.tan(lat)
    n = a / math.sqrt(1.0 - e2 * sl * sl)
    t = tl * tl
    c = ep2 * cl * cl
    aa = cl * (lon - lon0)
    m = a * ((1-e2/4-3*e2**2/64-5*e2**3/256)*lat
             -(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*lat)
             +(15*e2**2/256+45*e2**3/1024)*math.sin(4*lat)
             -(35*e2**3/3072)*math.sin(6*lat))
    easting = 500000.0 + k0*n*(aa + (1-t+c)*aa**3/6
                               + (5-18*t+t*t+72*c-58*ep2)*aa**5/120)
    northing = k0*(m + n*tl*(aa**2/2 + (5-t+9*c+4*c*c)*aa**4/24
                              + (61-58*t+t*t+600*c-330*ep2)*aa**6/720))
    if latitude_deg < 0:
        northing += 10000000.0
    return easting, northing


def projected_delta_enu(camera_llh: np.ndarray, target_llh: np.ndarray,
                        zone: int) -> np.ndarray:
    """Camera/antenna -> target delta in UTM East/North and height metres."""
    camera_e, camera_n = wgs84_to_utm(camera_llh[0], camera_llh[1], zone)
    target_e, target_n = wgs84_to_utm(target_llh[0], target_llh[1], zone)
    return np.asarray([target_e-camera_e, target_n-camera_n,
                       target_llh[2]-camera_llh[2]], dtype=float)


def antenna_to_camera_lever_enu(heading_deg: float, forward_m: float,
                                right_m: float, up_m: float) -> np.ndarray:
    """GNSS antenna -> camera lever arm expressed in ENU metres.

    Vehicle forward follows the recorded heading. This heading-only model
    assumes the vehicle is level; per-frame vehicle pitch/roll are unavailable.
    """
    heading = math.radians(heading_deg)
    forward = np.asarray([math.sin(heading), math.cos(heading), 0.0])
    right = np.asarray([math.cos(heading), -math.sin(heading), 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    return forward_m * forward + right_m * right + up_m * up


def world_to_camera(enu: np.ndarray, heading_deg: float, pitch_deg: float,
                    roll_deg: float, yaw_offset_deg: float) -> np.ndarray:
    """Rotate ENU into an x-right/y-down/z-forward camera frame."""
    yaw = math.radians(heading_deg + yaw_offset_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    # Initial level-camera axes expressed in ENU.
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.asarray([0.0, 0.0, -1.0])
    forward = np.asarray([math.sin(yaw), math.cos(yaw), 0.0])
    base = np.vstack([right, down, forward])
    # Camera-axis rotations: pitch about camera-right, roll about optical axis.
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    r_pitch = np.asarray([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r_roll = np.asarray([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return r_roll @ r_pitch @ base @ enu


def robust_line_fit(q: np.ndarray, pixel: np.ndarray, iterations: int = 10):
    """Fit pixel = focal*q + principal using Huber IRLS."""
    a = np.column_stack([q, np.ones_like(q)])
    weights = np.ones(len(q))
    beta = np.linalg.lstsq(a, pixel, rcond=None)[0]
    for _ in range(iterations):
        beta = np.linalg.lstsq(a * np.sqrt(weights[:, None]), pixel * np.sqrt(weights), rcond=None)[0]
        residual = pixel - a @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
    return beta, pixel - a @ beta, weights, np.linalg.cond(a)


def percentile_summary(x: np.ndarray) -> dict:
    return {"rmse": float(np.sqrt(np.mean(x * x))),
            "median_abs": float(np.median(np.abs(x))),
            "p95_abs": float(np.percentile(np.abs(x), 95)),
            "max_abs": float(np.max(np.abs(x)))}


def predict_pixels(data: np.ndarray, fx: float, fy: float,
                   cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Project rows [X,Y,Z,u,v] with the fitted pinhole intrinsics."""
    return fx * data[:, 0] / data[:, 2] + cx, fy * data[:, 1] / data[:, 2] + cy


def reprojection_metrics(data: np.ndarray, fx: float, fy: float,
                         cx: float, cy: float) -> dict:
    predicted_u, predicted_v = predict_pixels(data, fx, fy, cx, cy)
    residual_u = predicted_u - data[:, 3]
    residual_v = predicted_v - data[:, 4]
    radial = np.hypot(residual_u, residual_v)
    return {"count": len(data),
            "u_pixels": percentile_summary(residual_u),
            "v_pixels": percentile_summary(residual_v),
            "radial_pixels": percentile_summary(radial)}


def draw_dashed_line(draw, start, end, fill, width=4, dash=18, gap=12):
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2-x1, y2-y1)
    if not np.isfinite(length) or length <= 0:
        return
    ux, uy = (x2-x1)/length, (y2-y1)/length
    pos = 0.0
    while pos < length:
        stop = min(pos+dash, length)
        draw.line([(x1+ux*pos, y1+uy*pos),
                   (x1+ux*stop, y1+uy*stop)], fill=fill, width=width)
        pos += dash+gap


def draw_dataset_distribution(train: np.ndarray, validation: np.ndarray,
                              fx: float, fy: float, cx: float, cy: float,
                              width: int, height: int, output: Path,
                              point_radius: int) -> dict:
    """Draw original train pixels and validation observed/predicted pixels."""
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    for i in range(1, 8):
        x = round(i*width/8)
        draw.line([(x, 0), (x, height-1)], fill=(120, 130, 145, 55), width=2)
    for i in range(1, 6):
        y = round(i*height/6)
        draw.line([(0, y), (width-1, y)], fill=(120, 130, 145, 55), width=2)
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(40, 48, 62, 255), width=5)
    draw.line([(width//2, 0), (width//2, height-1)], fill=(50, 55, 65, 100), width=3)
    draw.line([(0, height//2), (width-1, height//2)], fill=(50, 55, 65, 100), width=3)

    train_fill, train_edge = (20, 105, 220, 185), (8, 55, 130, 255)
    val_fill, val_edge = (230, 55, 55, 220), (140, 15, 15, 255)
    predicted_colour, line_colour = (245, 145, 20, 255), (180, 80, 25, 175)
    outside = {"train": 0, "validation_observed": 0, "validation_predicted": 0}
    predicted_u, predicted_v = predict_pixels(validation, fx, fy, cx, cy)

    for row, up, vp in zip(validation, predicted_u, predicted_v):
        u, v = row[3], row[4]
        if not (np.isfinite(up) and np.isfinite(vp)):
            outside["validation_predicted"] += 1
            continue
        if not (0 <= up < width and 0 <= vp < height):
            outside["validation_predicted"] += 1
        clipped = (float(np.clip(up, -width, 2*width)),
                   float(np.clip(vp, -height, 2*height)))
        draw_dashed_line(draw, (float(u), float(v)), clipped, line_colour)

    for data, label, fill, edge in ((train, "train", train_fill, train_edge),
                                    (validation, "validation_observed", val_fill, val_edge)):
        for u, v in data[:, 3:5]:
            if not (0 <= u < width and 0 <= v < height):
                outside[label] += 1
                continue
            draw.ellipse((u-point_radius, v-point_radius, u+point_radius, v+point_radius),
                         fill=fill, outline=edge, width=3)

    cross = max(8, round(point_radius*1.3))
    for up, vp in zip(predicted_u, predicted_v):
        if np.isfinite(up) and np.isfinite(vp) and 0 <= up < width and 0 <= vp < height:
            draw.line([(up-cross, vp-cross), (up+cross, vp+cross)], fill=predicted_colour, width=5)
            draw.line([(up-cross, vp+cross), (up+cross, vp-cross)], fill=predicted_colour, width=5)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 42)
        small = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = small = ImageFont.load_default()
    draw.rounded_rectangle((35, 35, 890, 270), radius=18, fill=(255, 255, 255, 225),
                           outline=(60, 68, 80, 220), width=3)
    draw.ellipse((65, 65, 95, 95), fill=train_fill, outline=train_edge, width=3)
    draw.text((115, 52), f"Train original: {len(train)}", fill=(20, 25, 35, 255), font=font)
    draw.ellipse((65, 130, 95, 160), fill=val_fill, outline=val_edge, width=3)
    draw.text((115, 117), f"Validation original: {len(validation)}", fill=(20, 25, 35, 255), font=font)
    draw.line([(67, 201), (93, 227)], fill=predicted_colour, width=5)
    draw.line([(67, 227), (93, 201)], fill=predicted_colour, width=5)
    draw.text((115, 182), "Validation predicted", fill=(20, 25, 35, 255), font=font)
    draw.text((width-470, height-55), f"{width} x {height} pixels",
              fill=(40, 48, 62, 210), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    return {"path": str(output), "width": width, "height": height,
            "train_points": len(train), "validation_points": len(validation),
            "correspondence_lines": len(validation), "outside_image": outside}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file", type=Path)
    ap.add_argument("--camera-column", default="photo_wkt")
    ap.add_argument("--sign-column", default="sign_wkt")
    ap.add_argument("--heading-column", default="camera_heading")
    ap.add_argument("--u-column", default="u")
    ap.add_argument("--v-column", default="v")
    ap.add_argument("--id-column", default="id",
                    help="group column used to keep one sign in only one split")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera pitch in degrees; positive looks up")
    ap.add_argument("--roll", type=float, default=0.0, help="camera roll in degrees")
    ap.add_argument("--yaw-offset", type=float, default=0.0, help="camera optical-axis yaw minus recorded heading")
    ap.add_argument("--lever-forward", type=float, default=0.0,
                    help="GNSS antenna -> camera offset forward, metres")
    ap.add_argument("--lever-right", type=float, default=0.0,
                    help="GNSS antenna -> camera offset right, metres")
    ap.add_argument("--lever-up", type=float, default=0.0,
                    help="GNSS antenna -> camera offset up, metres")
    ap.add_argument("--utm-epsg", type=int,
                    help="override automatically selected WGS84 UTM EPSG")
    ap.add_argument("--min-depth", type=float, default=1.5)
    ap.add_argument("--min-height", type=float, default=0.5)
    ap.add_argument("--image-width", type=int, default=3840)
    ap.add_argument("--image-height", type=int, default=2880)
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distribution-image", type=Path,
                    default=Path("output/dataset_distribution.png"))
    ap.add_argument("--point-radius", type=int, default=10)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    samples, rejected, parsed = [], [], []
    with args.csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [args.id_column, args.camera_column, args.sign_column,
                    args.heading_column, args.u_column, args.v_column]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}; available: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            try:
                camera = parse_wkt_point_z(row[args.camera_column])
                sign = parse_wkt_point_z(row[args.sign_column])
                parsed.append((line_no, row[args.id_column], camera, sign,
                               float(row[args.heading_column]),
                               float(row[args.u_column]), float(row[args.v_column])))
            except Exception as exc:
                rejected.append({"line": line_no, "reason": str(exc)})

    if not parsed:
        raise ValueError("CSV contains no rows with valid WKT, heading and pixels")
    all_llh = np.vstack([x for item in parsed for x in (item[2], item[3])])
    centre_lon, centre_lat = np.mean(all_llh[:, :2], axis=0)
    selected_epsg = args.utm_epsg or automatic_utm_epsg(centre_lon, centre_lat)
    selected_zone = selected_epsg % 100
    selected_hemisphere = "north" if selected_epsg // 100 == 326 else "south"
    if selected_epsg not in range(32601, 32661) and selected_epsg not in range(32701, 32761):
        raise ValueError("--utm-epsg must be WGS84 UTM EPSG:32601..32660 or 32701..32760")
    if (centre_lat >= 0) != (selected_hemisphere == "north"):
        raise ValueError(f"EPSG:{selected_epsg} hemisphere does not match the data")
    selected_crs_name = f"WGS 84 / UTM zone {selected_zone}{'N' if selected_hemisphere == 'north' else 'S'}"
    zones_present = sorted({utm_zone(lon) for lon in all_llh[:, 0]})
    hemispheres_present = sorted({"north" if lat >= 0 else "south" for lat in all_llh[:, 1]})
    utm_warning = None
    if len(zones_present) > 1 or len(hemispheres_present) > 1:
        utm_warning = ("data spans multiple UTM zones/hemispheres; one CRS was selected "
                       f"for all rows: zones={zones_present}, hemispheres={hemispheres_present}")

    sample_ids = []
    for line_no, pid, camera, sign, heading, u, v in parsed:
        try:
                antenna_to_sign_enu = projected_delta_enu(camera, sign, selected_zone)
                lever_enu = antenna_to_camera_lever_enu(
                    heading, args.lever_forward, args.lever_right, args.lever_up)
                camera_to_sign_enu = antenna_to_sign_enu - lever_enu
                xyz = world_to_camera(camera_to_sign_enu, heading, args.pitch,
                                      args.roll, args.yaw_offset)
                if not np.all(np.isfinite([*xyz, u, v])) or xyz[2] <= args.min_depth or -xyz[1] <= args.min_height:
                    rejected.append({"line": line_no, "reason": f"invalid {pid}, X={xyz[0]:.3f} Y={xyz[1]:.3f} Z={xyz[2]:.3f}"})
                    continue
                samples.append((*xyz, u, v))
                sample_ids.append(pid)
        except Exception as exc:
            rejected.append({"line": line_no, "reason": str(exc)})

    if len(samples) < 4:
        raise ValueError(f"only {len(samples)} valid samples; at least 4 are required")
    data = np.asarray(samples, dtype=float)
    sample_ids = np.asarray(sample_ids)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    unique_ids = np.unique(sample_ids)
    if len(unique_ids) < 2:
        raise ValueError("at least two unique feature IDs are required for train/validation split")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_ids)
    validation_id_count = min(len(unique_ids)-1,
                              max(1, round(len(unique_ids)*args.validation_fraction)))
    validation_ids = set(unique_ids[:validation_id_count])
    validation_mask = np.asarray([value in validation_ids for value in sample_ids])
    train_data, validation_data = data[~validation_mask], data[validation_mask]
    if len(train_data) < 4 or len(validation_data) < 1:
        raise ValueError("train/validation split has too few rows")

    x, y, z, u, v = train_data.T
    bx, ru, wu, cond_x = robust_line_fit(x / z, u)
    by, rv, wv, cond_y = robust_line_fit(y / z, v)
    fx, cx = bx
    fy, cy = by
    distribution = draw_dataset_distribution(
        train_data, validation_data, fx, fy, cx, cy,
        args.image_width, args.image_height,
        args.distribution_image, args.point_radius)
    train_metrics = reprojection_metrics(train_data, fx, fy, cx, cy)
    validation_metrics = reprojection_metrics(validation_data, fx, fy, cx, cy)
    result = {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "valid_samples": len(samples), "rejected_samples": len(rejected),
        "split": {"method": "grouped by feature ID",
                  "id_column": args.id_column, "seed": args.seed,
                  "validation_fraction": args.validation_fraction,
                  "unique_feature_ids": len(unique_ids),
                  "training_feature_ids": len(unique_ids)-validation_id_count,
                  "validation_feature_ids": validation_id_count},
        "training": train_metrics,
        "validation": validation_metrics,
        "dataset_distribution": distribution,
        "assumptions": {"pitch_deg": args.pitch, "roll_deg": args.roll,
                        "yaw_offset_deg": args.yaw_offset, "heading": "clockwise from true north",
                        "lever_arm_antenna_to_camera_fru_m": {
                            "forward": args.lever_forward,
                            "right": args.lever_right,
                            "up": args.lever_up
                        },
                        "horizontal_coordinates": {
                            "method": "WGS84 UTM projected metres",
                            "epsg": selected_epsg,
                            "crs_name": selected_crs_name,
                            "selection_centre_lon_lat": [float(centre_lon), float(centre_lat)],
                            "utm_zones_present": zones_present
                        },
                        "camera_axes": "x right, y down, z forward", "distortion": "ignored"},
        "residual_u_pixels": percentile_summary(ru),
        "residual_v_pixels": percentile_summary(rv),
        "design_condition_number": {"u": float(cond_x), "v": float(cond_y)},
        "robust_inlier_like_count": int(np.sum((wu > 0.99) & (wv > 0.99))),
        "warnings": [],
    }
    if fx <= 0 or fy <= 0:
        result["warnings"].append("non-positive focal length: pose convention/attitude is likely wrong")
    if args.image_width and not (0 <= cx <= args.image_width):
        result["warnings"].append("cx lies outside the supplied image width")
    if args.image_height and not (0 <= cy <= args.image_height):
        result["warnings"].append("cy lies outside the supplied image height")
    if result["residual_u_pixels"]["rmse"] > 10 or result["residual_v_pixels"]["rmse"] > 10:
        result["warnings"].append("large reprojection residuals: supply accurate pitch/roll/yaw offset, synchronization, and distortion model")
    if utm_warning:
        result["warnings"].append(utm_warning)
    result["rejected_preview"] = rejected[:10]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)