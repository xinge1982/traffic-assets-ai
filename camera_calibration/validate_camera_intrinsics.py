#!/usr/bin/env python3
"""Validate supplied pinhole intrinsics using all camera/sign observations.

For every valid row, convert WGS84 coordinates to an automatically selected
UTM zone, apply a fixed antenna-to-camera lever arm and fixed installation
attitude, project the sign with supplied fx/fy/cx/cy, calculate pixel errors,
write per-point CSV/summary JSON, and draw observed-to-projected dashed lines.
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


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_wkt_point_z(text: str) -> np.ndarray:
    values = np.asarray([float(v) for v in NUMBER.findall(text)], dtype=float)
    if len(values) != 3:
        raise ValueError(f"expected POINT Z, got {text!r}")
    return values


def utm_zone(lon: float) -> int:
    return min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))


def automatic_utm_epsg(lon: float, lat: float) -> int:
    if not -80 <= lat <= 84:
        raise ValueError("UTM supports latitude from -80 to 84 degrees")
    return (32600 if lat >= 0 else 32700) + utm_zone(lon)


def wgs84_to_utm(lon_deg: float, lat_deg: float, zone: int) -> tuple[float, float]:
    """WGS84 to UTM easting/northing in metres."""
    a, f, k0 = 6378137.0, 1.0/298.257223563, 0.9996
    e2 = f*(2-f); ep2 = e2/(1-e2)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    lon0 = math.radians((zone-1)*6-180+3)
    s, c, t = math.sin(lat), math.cos(lat), math.tan(lat)
    n = a/math.sqrt(1-e2*s*s); tt = t*t; cc = ep2*c*c; aa = c*(lon-lon0)
    m = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*lat
           -(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*lat)
           +(15*e2**2/256+45*e2**3/1024)*math.sin(4*lat)
           -(35*e2**3/3072)*math.sin(6*lat))
    e = 500000+k0*n*(aa+(1-tt+cc)*aa**3/6
                     +(5-18*tt+tt**2+72*cc-58*ep2)*aa**5/120)
    no = k0*(m+n*t*(aa**2/2+(5-tt+9*cc+4*cc**2)*aa**4/24
                      +(61-58*tt+tt**2+600*cc-330*ep2)*aa**6/720))
    if lat_deg < 0: no += 10000000
    return e, no


def delta_utm(camera: np.ndarray, sign: np.ndarray, zone: int) -> np.ndarray:
    ce, cn = wgs84_to_utm(camera[0], camera[1], zone)
    se, sn = wgs84_to_utm(sign[0], sign[1], zone)
    return np.asarray([se-ce, sn-cn, sign[2]-camera[2]])


def lever_enu(heading_deg: float, forward: float, right: float, up: float) -> np.ndarray:
    h = math.radians(heading_deg)
    return np.asarray([forward*math.sin(h)+right*math.cos(h),
                       forward*math.cos(h)-right*math.sin(h), up])


def world_to_camera(enu: np.ndarray, heading_deg: float, yaw_offset: float,
                    pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(heading_deg+yaw_offset)
    base = np.asarray([[math.cos(yaw), -math.sin(yaw), 0],
                       [0, 0, -1],
                       [math.sin(yaw), math.cos(yaw), 0]]) @ enu
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    pitch = np.asarray([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    roll = np.asarray([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return roll @ pitch @ base


def summary(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "rmse": float(np.sqrt(np.mean(x*x))),
            "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x))}


def dashed_line(draw, a, b, fill, width=4, dash=18, gap=12):
    x1, y1 = a; x2, y2 = b; length = math.hypot(x2-x1, y2-y1)
    if not np.isfinite(length) or length == 0: return
    ux, uy = (x2-x1)/length, (y2-y1)/length
    p = 0.0
    while p < length:
        q = min(p+dash, length)
        draw.line([(x1+ux*p, y1+uy*p), (x1+ux*q, y1+uy*q)], fill=fill, width=width)
        p += dash+gap


def draw_plot(rows: list[dict], width: int, height: int, path: Path, radius: int) -> dict:
    image = Image.new("RGB", (width, height), (248, 250, 252)); draw = ImageDraw.Draw(image, "RGBA")
    for i in range(1, 8):
        x = round(i*width/8); draw.line([(x, 0), (x, height-1)], fill=(120,130,145,55), width=2)
    for i in range(1, 6):
        y = round(i*height/6); draw.line([(0, y), (width-1, y)], fill=(120,130,145,55), width=2)
    draw.rectangle((0, 0, width-1, height-1), outline=(40,48,62,255), width=5)
    draw.line([(width//2,0),(width//2,height-1)], fill=(50,55,65,100), width=3)
    draw.line([(0,height//2),(width-1,height//2)], fill=(50,55,65,100), width=3)
    observed, predicted, vector = (20,105,220,210), (245,145,20,255), (180,80,25,170)
    outside_observed = outside_predicted = 0
    for row in rows:
        u, v, up, vp = row["u"], row["v"], row["predicted_u"], row["predicted_v"]
        if not (0 <= u < width and 0 <= v < height): outside_observed += 1
        if not (0 <= up < width and 0 <= vp < height): outside_predicted += 1
        end = (float(np.clip(up, -width, 2*width)), float(np.clip(vp, -height, 2*height)))
        dashed_line(draw, (u, v), end, vector)
    for row in rows:
        u, v = row["u"], row["v"]
        if 0 <= u < width and 0 <= v < height:
            draw.ellipse((u-radius,v-radius,u+radius,v+radius), fill=observed,
                         outline=(8,55,130,255), width=3)
    cross = max(8, round(radius*1.3))
    for row in rows:
        u, v = row["predicted_u"], row["predicted_v"]
        if 0 <= u < width and 0 <= v < height:
            draw.line([(u-cross,v-cross),(u+cross,v+cross)], fill=predicted, width=5)
            draw.line([(u-cross,v+cross),(u+cross,v-cross)], fill=predicted, width=5)
    try:
        font=ImageFont.truetype("DejaVuSans.ttf",42); small=ImageFont.truetype("DejaVuSans.ttf",30)
    except OSError: font=small=ImageFont.load_default()
    draw.rounded_rectangle((35,35,800,205), radius=18, fill=(255,255,255,225), outline=(60,68,80,220), width=3)
    draw.ellipse((65,65,95,95), fill=observed, outline=(8,55,130,255), width=3)
    draw.text((115,52),f"Observed: {len(rows)}",fill=(20,25,35,255),font=font)
    draw.line([(67,136),(93,162)],fill=predicted,width=5); draw.line([(67,162),(93,136)],fill=predicted,width=5)
    draw.text((115,117),"Reprojected",fill=(20,25,35,255),font=font)
    draw.text((width-470,height-55),f"{width} x {height} pixels",fill=(40,48,62,210),font=small)
    path.parent.mkdir(parents=True,exist_ok=True); image.save(path,"PNG")
    return {"path":str(path),"correspondence_lines":len(rows),
            "outside_image":{"observed":outside_observed,"reprojected":outside_predicted}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file", type=Path)
    for name in ("fx","fy","cx","cy"):
        ap.add_argument(f"--{name}", type=float, required=True)
    ap.add_argument("--image-width",type=int,default=3840); ap.add_argument("--image-height",type=int,default=2880)
    ap.add_argument("--yaw-offset",type=float,default=0.0); ap.add_argument("--pitch",type=float,default=0.0); ap.add_argument("--roll",type=float,default=0.0)
    ap.add_argument("--lever-forward",type=float,default=0.0); ap.add_argument("--lever-right",type=float,default=0.0); ap.add_argument("--lever-up",type=float,default=0.0)
    ap.add_argument("--utm-epsg",type=int); ap.add_argument("--min-depth",type=float,default=0.5)
    ap.add_argument("--id-column",default="feature_id"); ap.add_argument("--camera-column",default="photo_wkt"); ap.add_argument("--sign-column",default="sign_wkt")
    ap.add_argument("--heading-column",default="camera_heading"); ap.add_argument("--u-column",default="u"); ap.add_argument("--v-column",default="v")
    ap.add_argument("--output-json",type=Path,default=Path("intrinsics_validation.json"))
    ap.add_argument("--output-csv",type=Path,default=Path("intrinsics_validation_points.csv"))
    ap.add_argument("--output-image",type=Path,default=Path("intrinsics_validation.png")); ap.add_argument("--point-radius",type=int,default=10)
    args=ap.parse_args()
    raw=[]; rejected=[]
    with args.csv_file.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f); required=[args.id_column,args.camera_column,args.sign_column,args.heading_column,args.u_column,args.v_column]
        missing=[x for x in required if x not in (reader.fieldnames or [])]
        if missing: raise ValueError(f"missing columns: {missing}")
        for line,row in enumerate(reader,2):
            try: raw.append((line,row[args.id_column],parse_wkt_point_z(row[args.camera_column]),parse_wkt_point_z(row[args.sign_column]),float(row[args.heading_column]),float(row[args.u_column]),float(row[args.v_column])))
            except Exception as exc: rejected.append({"line":line,"reason":str(exc)})
    if not raw: raise ValueError("no valid input rows")
    llh=np.vstack([p for row in raw for p in (row[2],row[3])]); lon,lat=np.mean(llh[:,:2],axis=0)
    epsg=args.utm_epsg or automatic_utm_epsg(lon,lat); zone=epsg%100
    if epsg not in range(32601,32661) and epsg not in range(32701,32761): raise ValueError("UTM EPSG must be 32601..32660 or 32701..32760")
    results=[]
    for line,fid,camera,sign,heading,u,v in raw:
        try:
            enu=delta_utm(camera,sign,zone)-lever_enu(heading,args.lever_forward,args.lever_right,args.lever_up)
            x,y,z=world_to_camera(enu,heading,args.yaw_offset,args.pitch,args.roll)
            if z<=args.min_depth: rejected.append({"line":line,"reason":f"behind camera, Z={z:.3f}"}); continue
            up=args.fx*x/z+args.cx; vp=args.fy*y/z+args.cy; du=up-u; dv=vp-v
            results.append({"line":line,"feature_id":fid,"X":x,"Y":y,"Z":z,"u":u,"v":v,"predicted_u":up,"predicted_v":vp,"error_u":du,"error_v":dv,"error_pixels":math.hypot(du,dv)})
        except Exception as exc: rejected.append({"line":line,"reason":str(exc)})
    if not results: raise ValueError("no points are valid and in front of camera")
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.output_csv.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    eu=np.asarray([r["error_u"] for r in results]); ev=np.asarray([r["error_v"] for r in results]); er=np.asarray([r["error_pixels"] for r in results])
    plot=draw_plot(results,args.image_width,args.image_height,args.output_image,args.point_radius)
    report={"intrinsics":{"fx":args.fx,"fy":args.fy,"cx":args.cx,"cy":args.cy},"image_size":[args.image_width,args.image_height],
            "extrinsics":{"yaw_offset_deg":args.yaw_offset,"pitch_deg":args.pitch,"roll_deg":args.roll,"lever_forward_right_up_m":[args.lever_forward,args.lever_right,args.lever_up]},
            "coordinates":{"utm_epsg":epsg,"zone":zone,"selection_centre_lon_lat":[float(lon),float(lat)]},
            "valid_points":len(results),"rejected_points":len(rejected),"error_u_pixels":summary(np.abs(eu)),"error_v_pixels":summary(np.abs(ev)),"radial_error_pixels":summary(er),
            "mean_error_pixels":float(np.mean(er)),"plot":plot,"per_point_csv":str(args.output_csv),"rejected_preview":rejected[:20]}
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc: print(f"ERROR: {exc}",file=sys.stderr); raise SystemExit(2)