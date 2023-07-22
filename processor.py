from typing import NamedTuple

import numpy as np
from skimage import color, draw, exposure, img_as_float, morphology, transform

from box import Box
from orto import fetch_orto
from polygon import Polygon
from transform_px_geo import transform_rad_to_px
from utils import save_image

FETCH_EXTEND = 8


class ProcessResult(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray


def make_polygon_mask(polygon: Polygon, image_box: Box, image_shape: tuple[int, int]) -> np.ndarray:
    antialiasing = 4
    image_shape = (image_shape[0] * antialiasing, image_shape[1] * antialiasing, image_shape[2])
    points = transform_rad_to_px(polygon, image_box, image_shape)
    mask = draw.polygon2mask(image_shape, points)
    mask = mask.astype(float)
    mask = transform.rescale(mask, 1 / antialiasing, channel_axis=2)
    return mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    hsv = color.rgb2hsv(image)
    s_channel = hsv[:, :, 1]
    p0 = np.min(s_channel)
    p100 = np.max(s_channel)
    s_channel = exposure.rescale_intensity(s_channel, in_range=(p0, p100))
    hsv[:, :, 1] = s_channel
    v_channel = hsv[:, :, 2]
    p05 = np.percentile(v_channel, 5)
    p95 = np.percentile(v_channel, 95)
    v_channel = exposure.rescale_intensity(v_channel, in_range=(p05, p95))
    hsv[:, :, 2] = v_channel
    return color.hsv2rgb(hsv)


def process_polygon(polygon: Polygon, raw_img: np.ndarray | None = None) -> ProcessResult:
    polygon_box = polygon.get_bounding_box()
    image_box = polygon_box.extend(meters=FETCH_EXTEND).squarify()

    if raw_img is None:
        raw_img = img_as_float(fetch_orto(image_box))

    orto_img = raw_img
    assert orto_img.shape[0] == orto_img.shape[1]
    assert orto_img.dtype == float
    save_image(orto_img, '1')

    mask_img = make_polygon_mask(polygon, image_box, orto_img.shape)
    save_image(mask_img, '2')

    orto_img = normalize_image(orto_img)
    save_image(orto_img, '3')

    mask_img_binary = mask_img[:, :, 0] > 0.5
    mask_img_contour = morphology.binary_dilation(mask_img_binary) ^ mask_img_binary

    overlay_img = np.copy(orto_img)
    overlay_img[mask_img_contour] = (1, 0, 0)
    save_image(overlay_img, '4')

    return ProcessResult(
        image=raw_img,
        mask=mask_img,
        overlay=overlay_img)
