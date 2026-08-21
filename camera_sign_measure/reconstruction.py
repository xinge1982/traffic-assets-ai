import numpy as np



def pixel_to_camera(
        u,
        v,
        depth,
        K
):

    fx=K[0,0]
    fy=K[1,1]

    cx=K[0,2]
    cy=K[1,2]


    X=(u-cx)*depth/fx

    Y=(v-cy)*depth/fy

    Z=depth


    return np.array(
        [
            X,
            Y,
            Z
        ]
    )



def corners_to_points(
        corners,
        depth,
        K
):

    points=[]


    for u,v in corners:


        z=depth[
            v,
            u
        ]


        p=pixel_to_camera(
            u,
            v,
            z,
            K
        )


        points.append(
            p
        )


    return np.array(points)