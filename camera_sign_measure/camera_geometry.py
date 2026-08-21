import math

import cv2
import numpy as np


def depth_curve(z):
    return 20.26 - 15.11 * math.log(z + 0.5)


def image_to_depth_point(
        u,
        v,
        image_width,
        image_height,
        depth_width,
        depth_height
):

    return (
        u * depth_width / image_width,
        v * depth_height / image_height
    )


def undistort_corners_to_depth(
        corners,
        K,
        camera_matrix,
        distortion_coefficients
):
    """
    Convert distorted source-image corner pixels to the undistorted
    DA3 depth/intrinsics pixel coordinate system.
    """
    corners = np.asarray(
        corners,
        dtype=np.float64
    ).reshape(-1, 1, 2)

    normalized = cv2.undistortPoints(
        corners,
        camera_matrix,
        distortion_coefficients
    ).reshape(-1, 2)

    depth_points = np.empty_like(normalized)

    depth_points[:, 0] = (
        normalized[:, 0] * K[0, 0] + K[0, 2]
    )
    depth_points[:, 1] = (
        normalized[:, 1] * K[1, 1] + K[1, 2]
    )

    return depth_points


def median_depth_inside_mask(
        depth,
        mask,
        erosion_size=7
):
    """
    Return the median valid depth inside an eroded sign mask.

    Erosion avoids sampling mixed sign/background values near the
    segmentation boundary. If erosion removes the entire region,
    fall back to the original mask.
    """
    mask = np.asarray(mask).squeeze().astype(np.uint8)

    if mask.shape != depth.shape:
        mask = cv2.resize(
            mask,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    kernel = np.ones(
        (erosion_size, erosion_size),
        dtype=np.uint8
    )

    inner_mask = cv2.erode(
        mask,
        kernel,
        iterations=1
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
        return None

    return float(np.median(depth[valid]))


def pixel_to_xyz(
        u,
        v,
        z,
        K
):

    fx = K[0,0]
    fy = K[1,1]

    cx = K[0,2]
    cy = K[1,2]


    X = (u-cx)*z/fx
    Y = (v-cy)*z/fy


    return np.array(
        [
            X,
            Y,
            z
        ],
        dtype=np.float32
    )


def calculate_sign_size_3d(
        corners,
        depth,
        mask,
        K,
        image_width,
        image_height,
        depth_width,
        depth_height,
        camera_matrix=None,
        distortion_coefficients=None
):
    raw_depth = median_depth_inside_mask(
        depth,
        mask
    )

    if raw_depth is None:
        return None

    scale_z = depth_curve(raw_depth)
    if scale_z < 0:
        scale_z = 0.001

    metric_depth = raw_depth * scale_z

    if (
        camera_matrix is not None
        and distortion_coefficients is not None
    ):
        depth_corners = undistort_corners_to_depth(
            corners,
            K,
            camera_matrix,
            distortion_coefficients
        )
    else:
        depth_corners = np.array(
            [
                image_to_depth_point(
                    p[0],
                    p[1],
                    image_width,
                    image_height,
                    depth_width,
                    depth_height
                )
                for p in corners
            ],
            dtype=np.float64
        )

    points=[]

    for u_depth, v_depth in depth_corners:
        xyz=pixel_to_xyz(
            u_depth,
            v_depth,
            metric_depth,
            K
        )

        points.append(
            xyz
        )

    points=np.array(points)

    top_width=np.linalg.norm(
        points[0]-points[1]
    )
    bottom_width=np.linalg.norm(
        points[3]-points[2]
    )
    left_height=np.linalg.norm(
        points[0]-points[3]
    )
    right_height=np.linalg.norm(
        points[1]-points[2]
    )

    width=(top_width+bottom_width)/2.0
    height=(left_height+right_height)/2.0

    return {
        "points":points,
        "raw_depth":float(raw_depth),
        "depth":float(metric_depth),
        "width":float(width),
        "height":float(height)
    }
