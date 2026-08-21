#!/usr/bin/env python3
"""Jointly estimate camera intrinsics and fixed vehicle-to-camera extrinsics.

Input CSV defaults match the supplied file: feature_id, camera_heading, u, v,
photo_wkt and sign_wkt. Coordinate conventions:
  * heading: degrees clockwise from true north
  * lever arm: GNSS antenna -> camera centre, vehicle Forward/Right/Up, metres
  * camera: x right, y down, z forward
  * installation pitch: positive means camera looks upward

This estimates a *constant* installation attitude. Unknown frame-to-frame road
pitch/roll cannot be recovered from heading-only input and remains model error.
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
from scipy.optimize import least_squares


NAMES = ("fx", "fy", "cx", "cy", "yaw_deg", "pitch_deg", "roll_deg",
         "lever_forward_m", "lever_right_m", "lever_up_m")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def wkt_xyz(text: str) -> np.ndarray:
    a = np.asarray([float(v) for v in NUM.findall(text)], dtype=float)
    if len(a) != 3:
        raise ValueError(f"not a POINT Z: {text!r}")
    return a


def llh_delta_enu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """WGS84 local camera/antenna-to-sign delta in ENU metres."""
    lon0, lat0, h0 = a
    lon1, lat1, h1 = b
    lat = math.radians((lat0 + lat1) / 2)
    ae, e2 = 6378137.0, 6.69437999014e-3
    n = ae / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    m = ae * (1 - e2) / (1 - e2 * math.sin(lat) ** 2) ** 1.5
    return np.array([math.radians(lon1-lon0)*n*math.cos(lat),
                     math.radians(lat1-lat0)*m, h1-h0])


def load_csv(path: Path, args) -> dict[str, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = [args.id_column, args.heading_column, args.u_column, args.v_column,
                  args.camera_column, args.sign_column]
        missing = [x for x in needed if x not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns {missing}; available={reader.fieldnames}")
        for line, r in enumerate(reader, 2):
            try:
                ant, sign = wkt_xyz(r[args.camera_column]), wkt_xyz(r[args.sign_column])
                values = (r[args.id_column], float(r[args.heading_column]),
                          float(r[args.u_column]), float(r[args.v_column]),
                          *llh_delta_enu(ant, sign))
                if np.all(np.isfinite(np.asarray(values[1:], float))):
                    rows.append(values)
            except Exception as exc:
                print(f"warning: skip CSV line {line}: {exc}", file=sys.stderr)
    if len(rows) < 20:
        raise ValueError(f"only {len(rows)} usable rows; at least 20 recommended")
    return {"id": np.asarray([x[0] for x in rows]),
            "heading": np.asarray([x[1] for x in rows], float),
            "u": np.asarray([x[2] for x in rows], float),
            "v": np.asarray([x[3] for x in rows], float),
            "enu": np.asarray([x[4:] for x in rows], float)}


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def project(p: np.ndarray, d: dict, min_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy, yaw0, pitch, roll, lf, lr, lu = p
    h = np.radians(d["heading"])
    # Antenna->camera lever arm expressed in ENU using road heading (level vehicle).
    lever = np.column_stack([lf*np.sin(h) + lr*np.cos(h),
                             lf*np.cos(h) - lr*np.sin(h),
                             np.full(len(h), lu)])
    q = d["enu"] - lever
    yaw = h + math.radians(yaw0)
    # Level camera coordinates after fixed yaw offset.
    x = q[:, 0]*np.cos(yaw) - q[:, 1]*np.sin(yaw)
    y = -q[:, 2]
    z = q[:, 0]*np.sin(yaw) + q[:, 1]*np.cos(yaw)
    # Pitch about camera-right axis, then roll about optical axis.
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    yp, zp = cp*y + sp*z, -sp*y + cp*z
    xr, yr = cr*x + sr*yp, -sr*x + cr*yp
    zs = np.maximum(zp, min_depth)
    return fx*xr/zs + cx, fy*yr/zs + cy, zp


def data_residual(p: np.ndarray, d: dict, min_depth: float) -> np.ndarray:
    uh, vh, z = project(p, d, min_depth)
    ru, rv = uh-d["u"], vh-d["v"]
    # Strong continuous penalty for points on/behind the image plane.
    bad = np.maximum(0.0, min_depth-z) * 1000.0
    return np.column_stack([ru + bad, rv + bad]).ravel()


def optimise_stage(p, active, train, lo, hi, min_depth, tie_f=False, max_nfev=2500):
    active = np.asarray(active, int)
    def unpack(x):
        q = p.copy(); q[active] = x
        if tie_f: q[1] = q[0]
        return q
    def fun(x):
        q = unpack(x)
        r = data_residual(q, train, min_depth)
        # Weak physical priors reduce drift in poorly observable configurations.
        priors = np.array([(q[2]-p[2])/300, (q[3]-p[3])/300,
                           q[6]/10, q[7]/5, q[8]/3, q[9]/3])
        return np.r_[r, priors]
    sol = least_squares(fun, p[active], bounds=(lo[active], hi[active]),
                        loss="huber", f_scale=5.0, x_scale="jac", max_nfev=max_nfev)
    return unpack(sol.x), sol


def staged_fit(initial, train, lo, hi, min_depth):
    p = initial.copy()
    # 1: square pixels + centred principal point; 2: extrinsics; 3: separate
    # focal lengths; 4: release principal point last.
    p, _ = optimise_stage(p, [0, 4, 5], train, lo, hi, min_depth, tie_f=True)
    p, _ = optimise_stage(p, [0, 4, 5, 6, 7, 8, 9], train, lo, hi, min_depth, tie_f=True)
    p, _ = optimise_stage(p, [0, 1, 4, 5, 6, 7, 8, 9], train, lo, hi, min_depth)
    p, sol = optimise_stage(p, range(10), train, lo, hi, min_depth, max_nfev=5000)
    return p, sol


def metrics(p, d, min_depth):
    uh, vh, z = project(p, d, min_depth)
    ru, rv = uh-d["u"], vh-d["v"]
    front = z > min_depth
    def one(x):
        x = x[front]
        return {"rmse": float(np.sqrt(np.mean(x*x))) if len(x) else None,
                "median_abs": float(np.median(np.abs(x))) if len(x) else None,
                "p95_abs": float(np.percentile(np.abs(x), 95)) if len(x) else None}
    return {"count": len(z), "front_count": int(front.sum()), "u_px": one(ru),
            "v_px": one(rv), "radial_px": one(np.hypot(ru, rv))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file", type=Path)
    ap.add_argument("--output", type=Path, default=Path("joint_calibration_result.json"))
    ap.add_argument("--image-width", type=int, default=3840)
    ap.add_argument("--image-height", type=int, default=2880)
    ap.add_argument("--camera-column", default="photo_wkt")
    ap.add_argument("--sign-column", default="sign_wkt")
    ap.add_argument("--heading-column", default="camera_heading")
    ap.add_argument("--u-column", default="u"); ap.add_argument("--v-column", default="v")
    ap.add_argument("--id-column", default="feature_id")
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multistart", type=int, default=12)
    ap.add_argument("--initial-focal", type=float, default=2000.0)
    ap.add_argument("--yaw-limit", type=float, default=90.0)
    ap.add_argument("--min-depth", type=float, default=0.5)
    args = ap.parse_args()
    d = load_csv(args.csv_file, args)
    rng = np.random.default_rng(args.seed)
    ids = np.unique(d["id"]); rng.shuffle(ids)
    nval = max(1, round(len(ids)*args.validation_fraction))
    val_ids = set(ids[:nval]); vm = np.asarray([x in val_ids for x in d["id"]])
    train, val = subset(d, ~vm), subset(d, vm)
    if len(train["id"]) < 20 or len(val["id"]) < 4:
        raise ValueError("train/validation split too small; adjust --validation-fraction")

    w, h = args.image_width, args.image_height
    lo = np.array([300, 300, w/2-300, h/2-300, -args.yaw_limit, -25, -15, -5, -3, -3.])
    hi = np.array([8000, 8000, w/2+300, h/2+300, args.yaw_limit, 25, 15, 5, 3, 3.])
    base = np.array([args.initial_focal, args.initial_focal, w/2, h/2, 0, 0, 0, 0, 0, 0.])
    candidates = []
    yaw_grid = np.linspace(-0.8*args.yaw_limit, 0.8*args.yaw_limit, args.multistart)
    for k, yaw in enumerate(yaw_grid):
        p0 = base.copy(); p0[4] = yaw
        if k: p0[0:2] *= rng.uniform(0.65, 1.5)
        try:
            p, sol = staged_fit(p0, train, lo, hi, args.min_depth)
            score = metrics(p, val, args.min_depth)["radial_px"]["median_abs"]
            candidates.append((float("inf") if score is None else score, p, sol))
        except Exception as exc:
            print(f"warning: start {k} failed: {exc}", file=sys.stderr)
    if not candidates: raise RuntimeError("all optimisation starts failed")
    candidates.sort(key=lambda x: x[0]); _, best, sol = candidates[0]
    span = hi-lo
    hit = [NAMES[i] for i in range(10)
           if (best[i]-lo[i])/span[i] < 0.005 or (hi[i]-best[i])/span[i] < 0.005]
    warnings = ["Only constant installation pitch/roll are estimated; per-frame road/vehicle attitude is unavailable.",
                "A low residual does not prove unique physical parameters; rerun with different seeds and compare."]
    if hit:
        warnings.append("Parameters at/near bounds (solution is not physically trustworthy): " + ", ".join(hit))
    val_metric = metrics(best, val, args.min_depth)
    if val_metric["radial_px"]["rmse"] is not None and val_metric["radial_px"]["rmse"] > 30:
        warnings.append("Validation radial RMSE exceeds 30 px: do not use this calibration as final camera parameters.")
    result = {
        "parameters": dict(zip(NAMES, map(float, best))),
        "image_size": [w, h], "train": metrics(best, train, args.min_depth),
        "validation": val_metric,
        "split": {"unique_signs": len(ids), "validation_signs": len(val_ids), "seed": args.seed},
        "optimizer": {"successful_starts": len(candidates), "requested_starts": args.multistart,
                      "success": bool(sol.success), "message": sol.message},
        "warnings": warnings,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
