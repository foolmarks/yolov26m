"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are
 proprietary to SiMa and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is
 strictly forbidden unless prior written permission is obtained from
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who
 have executed Confidentiality and Non-disclosure agreements explicitly
 covering such access.

 The copyright notice above does not evidence any actual or intended
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.

**************************************************************************
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def letterbox_resize(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: Union[int, Tuple[int, int, int]] = 0,
) -> np.ndarray:
    """
    Resize an image with letterbox padding while preserving aspect ratio.

    The function chooses a resize that fits inside `(target_h, target_w)` and
    pads symmetrically so left/right borders are equal and top/bottom borders
    are equal.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_h and target_w must be positive integers")

    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("image height and width must be non-zero")

    max_scale = min(target_w / src_w, target_h / src_h)
    ideal_w = max(1, min(target_w, int(round(src_w * max_scale))))
    ideal_h = max(1, min(target_h, int(round(src_h * max_scale))))

    candidates: List[Tuple[float, int, int, int]] = []

    def _try_candidate(resized_w: int, resized_h: int) -> None:
        if resized_w < 1 or resized_h < 1:
            return
        if resized_w > target_w or resized_h > target_h:
            return
        if (target_w - resized_w) % 2 != 0:
            return
        if (target_h - resized_h) % 2 != 0:
            return

        scale_w = resized_w / src_w
        scale_h = resized_h / src_h
        scale_mismatch = abs(scale_w - scale_h)
        area = resized_w * resized_h
        candidates.append((scale_mismatch, -area, resized_w, resized_h))

    start_w = ideal_w if ideal_w % 2 == target_w % 2 else ideal_w - 1
    if start_w < 1:
        start_w = 1 if target_w % 2 else 2
    for resized_w in range(start_w, 0, -2):
        scaled_h = int(round((resized_w / src_w) * src_h))
        _try_candidate(resized_w, scaled_h)

    start_h = ideal_h if ideal_h % 2 == target_h % 2 else ideal_h - 1
    if start_h < 1:
        start_h = 1 if target_h % 2 else 2
    for resized_h in range(start_h, 0, -2):
        scaled_w = int(round((resized_h / src_h) * src_w))
        _try_candidate(scaled_w, resized_h)

    if not candidates:
        raise ValueError(
            "Unable to produce symmetric letterbox padding for the requested "
            "target size without changing the image aspect ratio."
        )

    _, _, resized_w, resized_h = min(candidates)

    interpolation = (
        cv2.INTER_AREA
        if resized_w < src_w or resized_h < src_h
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = right = pad_w // 2
    top = bottom = pad_h // 2

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def means_sub(image: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Subtract per-channel means from a 3-channel image.

    The image is converted to float32 before subtraction so integer input
    types (for example uint8) do not underflow. If the means are provided as
    floating-point values and all are less than 1.0, image pixels are first
    normalized by dividing by 255.0. Input image shape can be `(H, W, 3)` or
    `(N, H, W, 3)` where `N` is batch size.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        pass
    elif image.ndim == 4 and image.shape[3] == 3:
        pass
    else:
        raise ValueError("image must have shape (H, W, 3) or (N, H, W, 3)")

    means_input = np.asarray(means)
    means_array = means_input.astype(np.float32, copy=False).reshape(-1)
    if means_array.size != 3:
        raise ValueError("means must contain exactly 3 elements")

    image_array = image.astype(np.float32, copy=False)
    if np.issubdtype(means_input.dtype, np.floating) and np.all(means_array < 1.0):
        image_array = image_array / 255.0

    reshape_dims = (1,) * (image_array.ndim - 1) + (3,)
    return image_array - means_array.reshape(reshape_dims)


def div_stddev(image: np.ndarray, stddev: np.ndarray) -> np.ndarray:
    """
    Divide a 3-channel image by per-channel standard deviation values.

    Input image shape can be `(H, W, 3)` or `(N, H, W, 3)` where `N` is batch
    size. If stddev is provided as floating-point values and all values are
    less than 1.0, image pixels are first normalized by dividing by 255.0.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        pass
    elif image.ndim == 4 and image.shape[3] == 3:
        pass
    else:
        raise ValueError("image must have shape (H, W, 3) or (N, H, W, 3)")

    stddev_input = np.asarray(stddev)
    stddev_array = stddev_input.astype(np.float32, copy=False).reshape(-1)
    if stddev_array.size != 3:
        raise ValueError("stddev must contain exactly 3 elements")
    if np.any(stddev_array == 0.0):
        raise ValueError("stddev values must be non-zero")

    image_array = image.astype(np.float32, copy=False)
    if np.issubdtype(stddev_input.dtype, np.floating) and np.all(stddev_array < 1.0):
        image_array = image_array / 255.0

    reshape_dims = (1,) * (image_array.ndim - 1) + (3,)
    return image_array / stddev_array.reshape(reshape_dims)


def div255(image: np.ndarray) -> np.ndarray:
    """
    Divide every pixel value in an image array by 255.0.
    """
    return image.astype(np.float32, copy=False) / 255.0


def shiftdiv(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixels from [0, 255] to [-1.0, 1.0].
    """
    return image.astype(np.float32, copy=False) / 127.5 - 1.0
