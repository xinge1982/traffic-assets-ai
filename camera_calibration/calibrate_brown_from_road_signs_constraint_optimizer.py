#!/usr/bin/env python3
"""Calibrate OpenCV K/D and fixed camera installation from road-sign data.

CSV defaults: feature_id, camera_heading, u, v, photo_wkt, sign_wkt.
The script groups by feature_id, fits on training signs, evaluates unseen signs,
and writes OpenCV-compatible K and D=[k1,k2,p1,p2,k3].

Camera axes: x right, y down, z forward. Heading: clockwise from true north.
Lever arm: GNSS antenna -> camera centre in vehicle Forward/Right/Up metres.
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
from scipy.optimize import least_squares

NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
         "yaw_offset_deg", "pitch_deg", "roll_deg",
         "lever_forward_m", "lever_right_m", "lever_up_m")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_point_z(text: str) -> np.ndarray:
    values = np.asarray([float(x) for x in NUM.findall(text)], dtype=float)
    if len(values) != 3:
        raise ValueError(f"expected POINT Z, got {text!r}")
    return values


def utm_zone(lon: float) -> int:
    return min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))


def automatic_utm_epsg(lon: float, lat: float) -> int:
    if not -80 <= lat <= 84:
        raise ValueError("automatic UTM requires latitude in [-80,84]")
    return (32600 if lat >= 0 else 32700) + utm_zone(lon)


def wgs84_to_utm(lon_deg: float, lat_deg: float, zone: int) -> tuple[float, float]:
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f);
    ep2 = e2 / (1 - e2)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    lon0 = math.radians((zone - 1) * 6 - 177)
    s, c, t = math.sin(lat), math.cos(lat), math.tan(lat)
    n = a / math.sqrt(1 - e2 * s * s);
    tt = t * t;
    cc = ep2 * c * c;
    aa = c * (lon - lon0)
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * lat))
    e = 500000 + k0 * n * (aa + (1 - tt + cc) * aa ** 3 / 6
                           + (5 - 18 * tt + tt ** 2 + 72 * cc - 58 * ep2) * aa ** 5 / 120)
    no = k0 * (m + n * t * (aa ** 2 / 2 + (5 - tt + 9 * cc + 4 * cc ** 2) * aa ** 4 / 24
                            + (61 - 58 * tt + tt ** 2 + 600 * cc - 330 * ep2) * aa ** 6 / 720))
    if lat_deg < 0: no += 10000000
    return e, no


def load_data(path: Path, args) -> tuple[dict, dict]:
    raw, rejected = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [args.id_column, args.camera_column, args.sign_column,
                    args.heading_column, args.u_column, args.v_column]
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing: raise ValueError(f"missing CSV columns: {missing}")
        for line, row in enumerate(reader, 2):
            try:
                raw.append((line, row[args.id_column],
                            parse_point_z(row[args.camera_column]),
                            parse_point_z(row[args.sign_column]),
                            float(row[args.heading_column]),
                            float(row[args.u_column]), float(row[args.v_column])))
            except Exception as exc:
                rejected.append({"line": line, "reason": str(exc)})
    if len(raw) < 20: raise ValueError("at least 20 valid rows are recommended")
    llh = np.vstack([x for r in raw for x in (r[2], r[3])])
    lon, lat = np.mean(llh[:, :2], axis=0)
    epsg = args.utm_epsg or automatic_utm_epsg(lon, lat);
    zone = epsg % 100
    if epsg not in range(32601, 32661) and epsg not in range(32701, 32761):
        raise ValueError("UTM EPSG must be 32601..32660 or 32701..32760")
    rows = []
    for line, fid, camera, sign, heading, u, v in raw:
        try:
            ce, cn = wgs84_to_utm(camera[0], camera[1], zone)
            se, sn = wgs84_to_utm(sign[0], sign[1], zone)
            rows.append((line, fid, heading, u, v, se - ce, sn - cn, sign[2] - camera[2]))
        except Exception as exc:
            rejected.append({"line": line, "reason": str(exc)})
    d = {"line": np.asarray([r[0] for r in rows]), "id": np.asarray([r[1] for r in rows]),
         "heading": np.asarray([r[2] for r in rows]), "u": np.asarray([r[3] for r in rows]),
         "v": np.asarray([r[4] for r in rows]), "enu": np.asarray([r[5:] for r in rows])}
    meta = {"utm_epsg": epsg, "utm_zone": zone, "selection_centre_lon_lat": [float(lon), float(lat)],
            "rejected_input": rejected}
    return d, meta


def subset(d, mask): return {k: v[mask] for k, v in d.items()}


def camera_xyz(p: np.ndarray, d: dict) -> np.ndarray:
    yaw0, pitch, roll, lf, lr, lu = p[9:15]
    h = np.radians(d["heading"])
    lever = np.column_stack([lf * np.sin(h) + lr * np.cos(h), lf * np.cos(h) - lr * np.sin(h), np.full(len(h), lu)])
    q = d["enu"] - lever;
    yaw = h + math.radians(yaw0)
    x = q[:, 0] * np.cos(yaw) - q[:, 1] * np.sin(yaw);
    y = -q[:, 2];
    z = q[:, 0] * np.sin(yaw) + q[:, 1] * np.cos(yaw)
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    yp = cp * y + sp * z;
    zp = -sp * y + cp * z
    return np.column_stack([cr * x + sr * yp, -sr * x + cr * yp, zp])


def project(p: np.ndarray, d: dict, min_depth: float):
    xyz = camera_xyz(p, d);
    x, y, z = xyz.T;
    zs = np.maximum(z, min_depth)
    xn = np.clip(x / zs, -10, 10);
    yn = np.clip(y / zs, -10, 10);
    r2 = xn * xn + yn * yn
    radial = 1 + p[4] * r2 + p[5] * r2 * r2 + p[8] * r2 * r2 * r2
    xd = xn * radial + 2 * p[6] * xn * yn + p[7] * (r2 + 2 * xn * xn)
    yd = yn * radial + p[6] * (r2 + 2 * yn * yn) + 2 * p[7] * xn * yn
    u = p[0] * xd + p[2];
    v = p[1] * yd + p[3]
    return np.nan_to_num(u, nan=0, posinf=1e6, neginf=-1e6), np.nan_to_num(v, nan=0, posinf=1e6, neginf=-1e6), z, xyz


def residual(p, d, min_depth, prior, width, height):
    up, vp, z, _ = project(p, d, min_depth)
    ru = np.clip(up - d["u"], -5000, 5000);
    rv = np.clip(vp - d["v"], -5000, 5000)
    bad = np.maximum(0, min_depth - z) * 1000
    data = np.column_stack([ru + bad, rv + bad]).ravel()
    # Weak physical priors prevent unobservable parameters from exploding.
    pr = np.asarray([(p[0] - p[1]) / 250, (p[2] - width / 2) / 150, (p[3] - height / 2) / 150,
                     p[4] / 0.35, p[5] / 0.20, p[6] / 0.02, p[7] / 0.02, p[8] / 0.10,
                     (p[9] - prior[9]) / 30, (p[10] - prior[10]) / 10, p[11] / 8,
                     (p[12] - prior[12]) / 3, (p[13] - prior[13]) / 2, (p[14] - prior[14]) / 2])
    return np.r_[data, pr]


def fit_stage(p, active, d, lo, hi, args, prior, tie_f=False):
    active = np.asarray(sorted(set(active)), int)
    if len(active) == 0: return p, None

    def unpack(x):
        q = p.copy();
        q[active] = x
        if tie_f: q[1] = q[0]
        return q

    sol = least_squares(lambda x: residual(unpack(x), d, args.min_depth, prior, args.image_width, args.image_height),
                        p[active], bounds=(lo[active], hi[active]), loss="huber", f_scale=5,
                        x_scale="jac", max_nfev=3500)
    return unpack(sol.x), sol


def staged_fit(p, train, lo, hi, args, prior, free_ext):
    ext = sorted(free_ext)
    p, _ = fit_stage(p, [0] + [i for i in (9, 10) if i in ext], train, lo, hi, args, prior, True)
    p, _ = fit_stage(p, [0] + ext, train, lo, hi, args, prior, True)
    base = [0, 2, 3] + ext if args.square_pixels else [0, 1, 2, 3] + ext
    p, _ = fit_stage(p, base, train, lo, hi, args, prior, args.square_pixels)
    p, _ = fit_stage(p, base + [4], train, lo, hi, args, prior, args.square_pixels)
    p, _ = fit_stage(p, base + [4, 5] + ([8] if args.optimize_k3 else []), train, lo, hi, args, prior,
                     args.square_pixels)
    full = base + [4, 5] + ([6, 7] if args.optimize_tangential else []) + ([8] if args.optimize_k3 else [])
    return fit_stage(p, full, train, lo, hi, args, prior, args.square_pixels)


def stats(p, d, min_depth):
    up, vp, z, _ = project(p, d, min_depth);
    front = z > min_depth
    ru = up[front] - d["u"][front];
    rv = vp[front] - d["v"][front];
    rr = np.hypot(ru, rv)

    def s(x): return {"rmse": float(np.sqrt(np.mean(x * x))), "mean_abs": float(np.mean(np.abs(x))),
                      "median_abs": float(np.median(np.abs(x))), "p95_abs": float(np.percentile(np.abs(x), 95))}

    return {"count": len(z), "front_count": int(front.sum()), "u_pixels": s(ru), "v_pixels": s(rv),
            "radial_pixels": s(rr)}


def dash(draw, a, b, fill, width=4, dash_len=18, gap=12):
    x1, y1 = a;
    x2, y2 = b;
    l = math.hypot(x2 - x1, y2 - y1)
    if not np.isfinite(l) or l == 0: return
    ux, uy = (x2 - x1) / l, (y2 - y1) / l;
    t = 0
    while t < l:
        q = min(t + dash_len, l);
        draw.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * q, y1 + uy * q)], fill=fill, width=width);
        t += dash_len + gap


def draw_plot(p, train, val, args):
    w, h = args.image_width, args.image_height;
    im = Image.new("RGB", (w, h), (248, 250, 252));
    dr = ImageDraw.Draw(im, "RGBA")
    for i in range(1, 8): x = round(i * w / 8);dr.line([(x, 0), (x, h - 1)], fill=(120, 130, 145, 55), width=2)
    for i in range(1, 6): y = round(i * h / 6);dr.line([(0, y), (w - 1, y)], fill=(120, 130, 145, 55), width=2)
    dr.rectangle((0, 0, w - 1, h - 1), outline=(40, 48, 62, 255), width=5);
    dr.line([(w // 2, 0), (w // 2, h)], fill=(50, 55, 65, 100), width=3);
    dr.line([(0, h // 2), (w, h // 2)], fill=(50, 55, 65, 100), width=3)
    up, vp, z, _ = project(p, val, args.min_depth);
    outside = 0
    for u, v, a, b, zz in zip(val["u"], val["v"], up, vp, z):
        if zz <= args.min_depth: continue
        if not (0 <= a < w and 0 <= b < h): outside += 1
        dash(dr, (u, v), (float(np.clip(a, -w, 2 * w)), float(np.clip(b, -h, 2 * h))), (180, 80, 25, 175))
    r = args.point_radius
    for u, v in zip(train["u"], train["v"]):
        if 0 <= u < w and 0 <= v < h: dr.ellipse((u - r, v - r, u + r, v + r), fill=(20, 105, 220, 185),
                                                 outline=(8, 55, 130, 255), width=3)
    for u, v in zip(val["u"], val["v"]):
        if 0 <= u < w and 0 <= v < h: dr.ellipse((u - r, v - r, u + r, v + r), fill=(230, 55, 55, 220),
                                                 outline=(140, 15, 15, 255), width=3)
    c = max(8, round(r * 1.3))
    for a, b, zz in zip(up, vp, z):
        if zz > args.min_depth and 0 <= a < w and 0 <= b < h:
            dr.line([(a - c, b - c), (a + c, b + c)], fill=(245, 145, 20, 255), width=5);
            dr.line([(a - c, b + c), (a + c, b - c)], fill=(245, 145, 20, 255), width=5)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    dr.rounded_rectangle((35, 35, 950, 270), 18, fill=(255, 255, 255, 225), outline=(60, 68, 80, 220), width=3)
    dr.text((70, 50), f"Train observed: {len(train['u'])}", fill=(20, 105, 220, 255), font=font)
    dr.text((70, 120), f"Validation observed: {len(val['u'])}", fill=(220, 45, 45, 255), font=font)
    dr.text((70, 190), "Validation projected: orange X + dashed error", fill=(190, 90, 15, 255), font=font)
    args.output_image.parent.mkdir(parents=True, exist_ok=True);
    im.save(args.output_image, "PNG")
    return {"path": str(args.output_image), "validation_predictions_outside_image": outside}


def main():
    ap = argparse.ArgumentParser(description=__doc__);
    ap.add_argument("csv_file", type=Path)
    ap.add_argument("--image-width", type=int, default=3840);
    ap.add_argument("--image-height", type=int, default=2880)
    ap.add_argument("--initial-focal", type=float, default=1400);
    ap.add_argument("--square-pixels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--yaw-offset", type=float, default=0);
    ap.add_argument("--pitch", type=float, default=0);
    ap.add_argument("--roll", type=float, default=0)
    ap.add_argument("--lever-forward", type=float, default=0);
    ap.add_argument("--lever-right", type=float, default=0);
    ap.add_argument("--lever-up", type=float, default=0)
    ap.add_argument("--lever-forward-min", type=float, default=-1.5,
                    help="lower bound for GNSS-to-camera forward offset in metres (default: -1.5)")
    ap.add_argument("--lever-forward-max", type=float, default=1.5,
                    help="upper bound for GNSS-to-camera forward offset in metres (default: 1.5)")
    ap.add_argument("--lever-right-min", type=float, default=-0.5,
                    help="lower bound for GNSS-to-camera right offset in metres (default: -0.5)")
    ap.add_argument("--lever-right-max", type=float, default=0.5,
                    help="upper bound for GNSS-to-camera right offset in metres (default: 0.5)")
    ap.add_argument("--lever-up-min", type=float, default=-1.0,
                    help="lower bound for GNSS-to-camera upward offset in metres (default: -1.0)")
    ap.add_argument("--lever-up-max", type=float, default=1.0,
                    help="upper bound for GNSS-to-camera upward offset in metres (default: 1.0)")
    ap.add_argument("--fix-extrinsics", action="store_true", help="keep supplied yaw/pitch/roll/lever fixed")
    ap.add_argument("--optimize-tangential", action="store_true");
    ap.add_argument("--optimize-k3", action="store_true")
    ap.add_argument("--validation-fraction", type=float, default=.25);
    ap.add_argument("--seed", type=int, default=42);
    ap.add_argument("--multistart", type=int, default=16)
    ap.add_argument("--min-depth", type=float, default=.5);
    ap.add_argument("--utm-epsg", type=int);
    ap.add_argument("--yaw-limit", type=float, default=180)
    ap.add_argument("--id-column", default="feature_id");
    ap.add_argument("--camera-column", default="photo_wkt");
    ap.add_argument("--sign-column", default="sign_wkt");
    ap.add_argument("--heading-column", default="camera_heading");
    ap.add_argument("--u-column", default="u");
    ap.add_argument("--v-column", default="v")
    ap.add_argument("--output-json", type=Path, default=Path("brown_calibration_result.json"));
    ap.add_argument("--output-csv", type=Path, default=Path("brown_calibration_points.csv"));
    ap.add_argument("--output-image", type=Path, default=Path("brown_calibration_validation.png"));
    ap.add_argument("--point-radius", type=int, default=10)
    args = ap.parse_args()
    lever_specs = (
        ("forward", args.lever_forward, args.lever_forward_min, args.lever_forward_max),
        ("right", args.lever_right, args.lever_right_min, args.lever_right_max),
        ("up", args.lever_up, args.lever_up_min, args.lever_up_max),
    )
    for axis, initial, lower, upper in lever_specs:
        if not lower < upper:
            ap.error(f"--lever-{axis}-min must be smaller than --lever-{axis}-max")
        if not lower <= initial <= upper:
            ap.error(f"--lever-{axis}={initial} is outside [{lower}, {upper}]")
    d, coord = load_data(args.csv_file, args)
    rng = np.random.default_rng(args.seed);
    ids = np.unique(d["id"]);
    rng.shuffle(ids);
    nv = max(1, min(len(ids) - 1, round(len(ids) * args.validation_fraction)));
    vid = set(ids[:nv]);
    vm = np.asarray([x in vid for x in d["id"]]);
    train, val = subset(d, ~vm), subset(d, vm)
    w, h = args.image_width, args.image_height
    base = np.asarray(
        [args.initial_focal, args.initial_focal, w / 2, h / 2, 0, 0, 0, 0, 0, args.yaw_offset, args.pitch, args.roll,
         args.lever_forward, args.lever_right, args.lever_up], float);
    prior = base.copy()
    lo = np.asarray([300, 300, w / 2 - 300, h / 2 - 300, -1, -1, -.1, -.1, -1, -args.yaw_limit, -25, -15,
                     args.lever_forward_min, args.lever_right_min, args.lever_up_min], float)
    hi = np.asarray([6000, 6000, w / 2 + 300, h / 2 + 300, 1, 1, .1, .1, 1, args.yaw_limit, 25, 15,
                     args.lever_forward_max, args.lever_right_max, args.lever_up_max], float)
    free_ext = [] if args.fix_extrinsics else list(range(9, 15));
    candidates = []
    yaw_starts = [args.yaw_offset] if args.fix_extrinsics else np.linspace(-.8 * args.yaw_limit, .8 * args.yaw_limit,
                                                                           args.multistart)
    for yaw in yaw_starts:
        p0 = base.copy();
        p0[9] = yaw
        try:
            p, sol = staged_fit(p0, train, lo, hi, args, prior, free_ext);
            m = stats(p, val, args.min_depth);
            score = m["radial_pixels"]["median_abs"] if m["front_count"] >= max(4, .8 * len(val["u"])) else float(
                "inf");
            candidates.append((score, p, sol))
        except Exception as exc:
            print(f"warning: start failed: {exc}", file=sys.stderr)
    if not candidates: raise RuntimeError("all optimization starts failed")
    candidates.sort(key=lambda x: x[0]);
    score, best, sol = candidates[0];
    plot = draw_plot(best, train, val, args)
    # Per-point output for independent inspection.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["split", "line", "feature_id", "u", "v", "projected_u", "projected_v", "error_u", "error_v",
                  "radial_error", "X", "Y", "Z"];
        wr = csv.DictWriter(f, fieldnames=fields);
        wr.writeheader()
        for label, dd in (("train", train), ("validation", val)):
            up, vp, z, xyz = project(best, dd, args.min_depth)
            for i in range(len(z)):
                wr.writerow(dict(zip(fields,
                                     [label, int(dd["line"][i]), dd["id"][i], dd["u"][i], dd["v"][i], up[i], vp[i],
                                      up[i] - dd["u"][i], vp[i] - dd["v"][i],
                                      math.hypot(up[i] - dd["u"][i], vp[i] - dd["v"][i]), *xyz[i]])))
    span = hi - lo;
    hit = [NAMES[i] for i in range(15) if (best[i] - lo[i]) / span[i] < .005 or (hi[i] - best[i]) / span[i] < .005]
    warnings = []
    if hit: warnings.append("parameters at/near bounds: " + ", ".join(hit))
    vmtr = stats(best, val, args.min_depth)
    if vmtr["radial_pixels"]["rmse"] > 30: warnings.append(
        "validation RMSE >30 px; do not treat as final physical calibration")
    warnings.append("heading-only data cannot model frame-varying road pitch/roll or DJI stabilization")
    K = [[float(best[0]), 0, float(best[2])], [0, float(best[1]), float(best[3])], [0, 0, 1]];
    D = [float(best[i]) for i in (4, 5, 6, 7, 8)]
    out = {"opencv": {"camera_matrix_K": K, "dist_coeffs_D_k1_k2_p1_p2_k3": D},
           "parameters": dict(zip(NAMES, map(float, best))),
           "parameter_bounds": {"lever_forward_m": [args.lever_forward_min, args.lever_forward_max],
                                "lever_right_m": [args.lever_right_min, args.lever_right_max],
                                "lever_up_m": [args.lever_up_min, args.lever_up_max]},
           "coordinates": {k: v for k, v in coord.items() if k != "rejected_input"},
           "split": {"unique_signs": len(ids), "train_rows": len(train["u"]), "validation_rows": len(val["u"]),
                     "validation_signs": nv, "seed": args.seed}, "train": stats(best, train, args.min_depth),
           "validation": vmtr, "plot": plot,
           "optimizer": {"success": bool(sol.success) if sol else True, "message": sol.message if sol else "fixed",
                         "successful_starts": len(candidates)}, "warnings": warnings,
           "rejected_input_preview": coord["rejected_input"][:10]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True);
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8");
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr);raise SystemExit(2)
