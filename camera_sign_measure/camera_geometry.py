import numpy as np
import math

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
        K,
        image_width,
        image_height,
        depth_width,
        depth_height
):
    points=[]

    for p in corners:

        # Convert source-image coordinates to the actual DA3 depth/K
        # coordinate system instead of assuming a fixed 504 x 504 size.
        u_depth,v_depth=image_to_depth_point(
            p[0],
            p[1],
            image_width,
            image_height,
            depth_width,
            depth_height
        )

        z=depth[
            p[1],
            p[0]
        ]

        scale_z = depth_curve(z)
        if scale_z < 0:
            scale_z = 0.001

        xyz=pixel_to_xyz(
            u_depth,
            v_depth,
            z * scale_z,
            K
        )

        points.append(
            xyz
        )

    points=np.array(points)

    width=np.linalg.norm(
        points[0]-points[1]
    )

    height=np.linalg.norm(
        points[1]-points[2]
    )

    return {
        "points":points,
        "width":float(width),
        "height":float(height)
    }