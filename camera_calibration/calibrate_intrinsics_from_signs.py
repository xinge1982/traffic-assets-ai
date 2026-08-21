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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file", type=Path)
    ap.add_argument("--camera-column", default="photo_wkt")
    ap.add_argument("--sign-column", default="sign_wkt")
    ap.add_argument("--heading-column", default="camera_heading")
    ap.add_argument("--u-column", default="u")
    ap.add_argument("--v-column", default="v")
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
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--image-width", type=int, help="used only for plausibility checks")
    ap.add_argument("--image-height", type=int, help="used only for plausibility checks")
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    samples, rejected, parsed = [], [], []
    with args.csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [args.camera_column, args.sign_column, args.heading_column, args.u_column, args.v_column]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}; available: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            try:
                camera = parse_wkt_point_z(row[args.camera_column])
                sign = parse_wkt_point_z(row[args.sign_column])
                parsed.append((line_no, camera, sign,
                               float(row[args.heading_column]),
                               float(row[args.u_column]), float(row[args.v_column])))
            except Exception as exc:
                rejected.append({"line": line_no, "reason": str(exc)})

    if not parsed:
        raise ValueError("CSV contains no rows with valid WKT, heading and pixels")
    all_llh = np.vstack([x for item in parsed for x in (item[1], item[2])])
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

    for line_no, camera, sign, heading, u, v in parsed:
        try:
                antenna_to_sign_enu = projected_delta_enu(camera, sign, selected_zone)
                lever_enu = antenna_to_camera_lever_enu(
                    heading, args.lever_forward, args.lever_right, args.lever_up)
                camera_to_sign_enu = antenna_to_sign_enu - lever_enu
                xyz = world_to_camera(camera_to_sign_enu, heading, args.pitch,
                                      args.roll, args.yaw_offset)
                if not np.all(np.isfinite([*xyz, u, v])) or xyz[2] <= args.min_depth:
                    rejected.append({"line": line_no, "reason": f"invalid/behind camera, Z={xyz[2]:.3f}"})
                    continue
                samples.append((*xyz, u, v))
        except Exception as exc:
            rejected.append({"line": line_no, "reason": str(exc)})

    if len(samples) < 4:
        raise ValueError(f"only {len(samples)} valid samples; at least 4 are required")
    data = np.asarray(samples)
    x, y, z, u, v = data.T
    bx, ru, wu, cond_x = robust_line_fit(x / z, u)
    by, rv, wv, cond_y = robust_line_fit(y / z, v)
    fx, cx = bx
    fy, cy = by
    result = {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "valid_samples": len(samples), "rejected_samples": len(rejected),
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
