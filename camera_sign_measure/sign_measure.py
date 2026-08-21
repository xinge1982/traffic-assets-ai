import numpy as np

DEPTH_SCALE = 4.6

def distance_2d(p1, p2):

    return np.linalg.norm(
        np.array(p1) -
        np.array(p2)
    )



def calculate_sign_size(
        corners,
        depth,
        K
):
    """
    corners:

    [
      top-left,
      top-right,
      bottom-right,
      bottom-left
    ]

    depth:
        Depth Anything V3 output

    K:
        camera intrinsic matrix

    return:
        width meter
        height meter
    """


    fx = K[0,0]
    fy = K[1,1]


    #
    # sign center
    #
    cx = int(
        np.mean(
            corners[:,0]
        )
    )

    cy = int(
        np.mean(
            corners[:,1]
        )
    )


    #
    # depth at center
    #
    z = depth[cy, cx] * DEPTH_SCALE


    #
    # pixel length
    #
    pixel_width = distance_2d(
        corners[0],
        corners[1]
    )


    pixel_height = distance_2d(
        corners[1],
        corners[2]
    )


    #
    # perspective projection
    #
    width_meter = (
        pixel_width *
        z /
        fx
    )


    height_meter = (
        pixel_height *
        z /
        fy
    )


    return {

        "pixel_width":
            pixel_width,

        "pixel_height":
            pixel_height,

        "depth":
            float(z),

        "width":
            float(width_meter),

        "height":
            float(height_meter),

        "center":
            [
                cx,
                cy
            ]
    }