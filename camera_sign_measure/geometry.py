import cv2
import numpy as np

def mask_to_contour_quad(
        mask,
        epsilon_ratio=0.02
):

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )


    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours)==0:
        return None


    contour=max(
        contours,
        key=cv2.contourArea
    )


    if cv2.contourArea(contour) < 50:
        return None


    perimeter=cv2.arcLength(
        contour,
        True
    )


    approx=cv2.approxPolyDP(
        contour,
        epsilon_ratio * perimeter,
        True
    )


    points=approx.reshape(-1,2)


    if len(points)!=4:

        rect=cv2.minAreaRect(
            contour
        )

        points=np.int32(
            cv2.boxPoints(rect)
        )


    return order_points(points)

def order_points(pts):

    """
    return:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    rect=np.zeros(
        (4,2),
        dtype=np.float32
    )


    s=pts.sum(axis=1)


    rect[0]=pts[
        np.argmin(s)
    ]

    rect[2]=pts[
        np.argmax(s)
    ]


    diff=np.diff(
        pts,
        axis=1
    )


    rect[1]=pts[
        np.argmin(diff)
    ]

    rect[3]=pts[
        np.argmax(diff)
    ]


    return rect.astype(
        np.int32
    )