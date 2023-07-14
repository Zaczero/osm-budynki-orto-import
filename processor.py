from itertools import chain, pairwise
from operator import itemgetter
from typing import Sequence

import numpy as np
from shapely.geometry import LineString
from skimage import (color, draw, exposure, feature, filters, img_as_float,
                     morphology, transform)

from config import SEED
from merge_lines import merge_lines
from orto import fetch_orto
from polygon import Polygon
from score_lines import score_lines
from transform_px_geo import transform_geo_to_px
from utils import random_color, save_image

FETCH_EXTEND = 8
ROOF_EXTEND = 1

NOISE_PENALTY = (
    (0.00, 0.00),
    (0.30, 0.00),
    (0.35, 0.10),
    (0.40, 0.30),
    (0.50, 0.50),
    (0.60, 1.00),
    (1.00, 1.00),
)

CONTRAST_PENALTY = (
    (0.00, 1.00),
    (0.06, 0.50),
    (0.07, 0.30),
    (0.08, 0.10),
    (0.09, 0.00),
    (1.00, 0.00),
)


def penalty(value: float, config: Sequence) -> float:
    for (x1, y1), (x2, y2) in pairwise(config):
        if x1 <= value <= x2:
            return y1 + (y2 - y1) * (value - x1) / (x2 - x1)


def get_polygon_mask(polygon_lines: Sequence[tuple[tuple[float, float], tuple[float, float]]], shape: tuple[int, int]) -> np.ndarray:
    polygon_points = tuple(map(itemgetter(0), polygon_lines))
    rr, cc = draw.polygon(*zip(*polygon_points), shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def process_polygon(polygon: Polygon) -> float:
    polygon_box = polygon.get_bounding_box()
    orto_box = polygon_box.extend(meters=FETCH_EXTEND)
    orto_img = fetch_orto(orto_box)
    save_image(orto_img, '1')

    # boost contrast
    # orto_img = exposure.equalize_adapthist(orto_img, clip_limit=0.006)
    orto_img = exposure.rescale_intensity(orto_img, in_range=(0.25, 0.75))
    save_image(orto_img, '2')

    # median filter
    orto_img = filters.median(orto_img, morphology.square(7))
    save_image(orto_img, '3')

    edges = feature.canny(orto_img)
    # edges = morphology.dilation(edges, morphology.disk(3))
    # edges = morphology.erosion(edges, morphology.disk(3))
    edges = morphology.dilation(edges, morphology.disk(1))
    save_image(edges, '4')

    # count noise pixels
    polygon_lines = transform_geo_to_px(tuple(e for e in pairwise(polygon.points)), orto_box, orto_img.shape)
    polygon_mask = get_polygon_mask(polygon_lines, orto_img.shape)
    polygon_mask = morphology.erosion(polygon_mask, morphology.disk(3))
    polygon_mask_count = np.count_nonzero(polygon_mask)

    if not polygon_mask_count:
        print('[PROCESS] Polygon mask is empty')
        return 0

    noise_perc = np.count_nonzero(edges & polygon_mask) / polygon_mask_count
    noise_penalty_ = penalty(noise_perc, NOISE_PENALTY)

    # measure contrast in mask
    contrast = np.std(orto_img[polygon_mask])
    contrast_penalty = penalty(contrast, CONTRAST_PENALTY)

    lines = chain(
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 1),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 2),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 3))
    lines = tuple(((p0[1], p0[0]), (p1[1], p1[0])) for p0, p1 in lines)  # fix x, y order
    lines_img = img_as_float(color.gray2rgb(np.copy(edges)))
    lines_img_2 = np.zeros_like(lines_img)

    for line_points in lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        lines_img[rr, cc] = lines_img_2[rr, cc] = random_color()

    save_image(lines_img, '5')
    save_image(lines_img_2, '6')

    merged_lines = merge_lines(lines, dist_threshold=2, angle_threshold=10)
    merged_lines_img = np.zeros_like(lines_img)

    for line_points in merged_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        merged_lines_img[rr, cc] = random_color()

    save_image(merged_lines_img, '7')

    long_lines = tuple(line for line in merged_lines if LineString(line).length > 20)
    long_lines_img = np.zeros_like(lines_img)

    for line_points in long_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        long_lines_img[rr, cc] = random_color()

    save_image(long_lines_img, '8')

    polygon_img = np.copy(long_lines_img)

    for line_points in polygon_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        polygon_img[rr, cc] = np.array([1, 0, 0])

    save_image(polygon_img, '9')

    score = score_lines(polygon_lines, long_lines)

    if noise_penalty_ > 0:
        score_prev = score
        score *= (1 - noise_penalty_)
        print(f'[PROCESS][{noise_perc:05.3f}] ☔ Image is noisy, score reduced {score_prev:05.3f} ➡️ {score:05.3f}')

    if contrast_penalty > 0:
        score_prev = score
        score *= (1 - contrast_penalty)
        print(f'[PROCESS][{contrast:05.3f}] 🎨 Image is low contrast, score reduced {score_prev:05.3f} ➡️ {score:05.3f}')

    return score
