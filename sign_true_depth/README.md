# Sign true-depth training data

`generate_training_data.py` creates calibration samples for replacing the
current `depth_curve()` function.

## Input

The default input file is:

```text
sign_true_depth/input/camera_sign.csv
```

Required columns:

```text
id,img,x1,y1,x2,y2,camera_heading,u,v,photo_wkt,sign_wkt
```

The value of `img` is appended to `--image-url-prefix`.

## Run

Run from the repository root:

```bash
python sign_true_depth/generate_training_data.py \
  --image-url-prefix "https://example.com/photos/"
```

Example with camera installation offsets:

```bash
python sign_true_depth/generate_training_data.py \
  --image-url-prefix "https://example.com/photos/" \
  --yaw-offset 0.0 \
  --pitch-offset 0.0 \
  --roll-offset 0.0 \
  --lever-forward 0.0 \
  --lever-right 0.0 \
  --lever-up 0.0
```

Use `--limit 20` for a small initial test.

## Processing

For each unique image, the script:

1. downloads and caches the source image;
2. runs Depth Anything V3 once;
3. initializes the image in SAM2 once;
4. segments every CSV YOLO bounding box;
5. extracts four ordered sign corners;
6. resizes DA3 depth to source-image resolution;
7. erodes the sign mask with the same 7 x 7 kernel used by
   `median_depth_inside_mask()`;
8. calculates raw-depth statistics;
9. converts `photo_wkt` and `sign_wkt` from WGS84 through ECEF/ENU into
   camera coordinates;
10. writes the joined training sample.

Camera coordinates are:

```text
+X = camera right
+Y = camera down
+Z = camera forward
```

The target used to train a replacement for `depth_curve()` is:

```text
true_camera_z
```

The compatibility scale target is:

```text
target_depth_scale = true_camera_z / raw_depth_median
```

## Output

Default output directory:

```text
sign_true_depth/output/
├── images/                 downloaded image cache
├── depth/                  native DA3 .npy depth arrays
├── masks/                  one SAM2 mask per detection
├── training_data.csv       successful samples
└── failed_samples.csv      failed and behind-camera samples
```

Both CSV files are rewritten after each completed image, so an interrupted
long run retains the completed results.

## Important pose assumptions

`camera_heading` is treated as clockwise degrees from true north. The
default camera installation offsets and GNSS-to-camera lever arms are zero.

For reliable `true_camera_z` labels, pass measured camera installation
offsets. A GNSS antenna position is not automatically the camera optical
center.
