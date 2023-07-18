from itertools import chain, pairwise
from operator import itemgetter
from typing import Sequence

import numpy as np
from scipy.stats import kurtosis, skew
from shapely.geometry import LineString
from skimage import (color, draw, exposure, feature, img_as_float,
                     img_as_ubyte, morphology, restoration, transform)
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.cluster import KMeans

from config import SEED
from merge_lines import merge_lines
from orto import fetch_orto
from polygon import Polygon
from score_lines import score_lines
from transform_px_geo import transform_geo_to_px
from utils import random_color, save_image

FETCH_EXTEND = 8
ROOF_EXTEND = 1


def get_polygon_mask(polygon_lines: Sequence[tuple[tuple[float, float], tuple[float, float]]], shape: tuple[int, int]) -> np.ndarray:
    polygon_points = tuple(map(itemgetter(0), polygon_lines))
    rr, cc = draw.polygon(*zip(*polygon_points), shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def process_polygon(polygon: Polygon, orto_img: np.ndarray | None = None) -> dict | None:
    features = {}
    polygon_box = polygon.get_bounding_box()
    orto_box = polygon_box.extend(meters=FETCH_EXTEND)

    if orto_img is None:
        orto_img = fetch_orto(orto_box)

    assert orto_img.dtype == float
    save_image(orto_img, '1')

    polygon_lines = transform_geo_to_px(tuple(e for e in pairwise(polygon.points)), orto_box, orto_img.shape)
    polygon_mask = get_polygon_mask(polygon_lines, orto_img.shape)
    non_polygon_mask = ~polygon_mask
    polygon_mask = morphology.erosion(polygon_mask, morphology.disk(3))
    non_polygon_mask = morphology.erosion(non_polygon_mask, morphology.disk(3))
    polygon_mask_count = np.count_nonzero(polygon_mask)
    non_polygon_mask_count = np.count_nonzero(non_polygon_mask)

    if not polygon_mask_count or not non_polygon_mask_count:
        print('[PROCESS] Polygon mask is empty')
        return None

    # contrast stretching
    p05 = np.percentile(orto_img, 5)
    p95 = np.percentile(orto_img, 95)
    orto_img = exposure.rescale_intensity(orto_img, in_range=(p05, p95))
    save_image(orto_img, '2')

    no_mask = np.ones_like(polygon_mask)

    for mask, prefix in zip((no_mask, polygon_mask, non_polygon_mask), ('', 'inside_', 'outside_')):
        masked = img_as_ubyte(orto_img)
        masked[~mask] = 0
        glcm = graycomatrix(masked, (1,), (0, np.pi/4, np.pi/2, 3*np.pi/4), levels=256, symmetric=True, normed=True)

        for prop in ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'):
            features[f'{prefix}{prop}'] = graycoprops(glcm, prop)[0, 0]

        hog_features: np.ndarray = feature.hog(masked,
                                               orientations=8,
                                               pixels_per_cell=(16, 16),
                                               cells_per_block=(1, 1),
                                               visualize=False)

        features[f'{prefix}hog_entropy'] = shannon_entropy(hog_features)
        features[f'{prefix}hog_mean'] = hog_features.mean()
        features[f'{prefix}hog_var'] = hog_features.var()
        features[f'{prefix}hog_skew'] = skew(hog_features)
        features[f'{prefix}hog_kurtosis'] = kurtosis(hog_features)

        masked = np.ma.masked_array(orto_img, mask=~mask)
        masked_compressed = masked.compressed()
        features[f'{prefix}entropy'] = shannon_entropy(masked)
        features[f'{prefix}mean'] = masked.mean()
        features[f'{prefix}var'] = masked.var()
        features[f'{prefix}skew'] = skew(masked_compressed)
        features[f'{prefix}kurtosis'] = kurtosis(masked_compressed)

    # denoise
    orto_img_denoise = restoration.denoise_nl_means(orto_img)
    save_image(orto_img_denoise, '3')

    # edge detection
    edges = feature.canny(orto_img_denoise)
    # edges = morphology.dilation(edges, morphology.disk(3))
    # edges = morphology.erosion(edges, morphology.disk(3))
    edges = morphology.dilation(edges, morphology.disk(1))
    save_image(edges, '4')

    # measure edge noise
    features['in_noise'] = np.count_nonzero(edges & polygon_mask) / polygon_mask_count
    features['out_noise'] = np.count_nonzero(edges & non_polygon_mask) / non_polygon_mask_count

    lines = chain(
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 1),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 2),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 3))
    lines = tuple(((p0[1], p0[0]), (p1[1], p1[0])) for p0, p1 in lines)  # fix x, y order
    lines_edges_img = img_as_float(color.gray2rgb(np.copy(edges)))
    lines_img = np.zeros_like(lines_edges_img)

    for line_points in lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        lines_edges_img[rr, cc] = lines_img[rr, cc] = random_color()

    save_image(lines_edges_img, '5')
    save_image(lines_img, '6')

    merged_lines = merge_lines(lines, dist_threshold=2, angle_threshold=10)
    merged_lines_img = np.zeros_like(lines_edges_img)

    for line_points in merged_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        merged_lines_img[rr, cc] = random_color()

    save_image(merged_lines_img, '7')

    long_lines = tuple(line for line in merged_lines if LineString(line).length > 20)
    long_lines_img = np.zeros_like(lines_edges_img)

    for line_points in long_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        long_lines_img[rr, cc] = random_color()

    save_image(long_lines_img, '8')

    polygon_img = np.copy(long_lines_img)

    for line_points in polygon_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        polygon_img[rr, cc] = np.array([1, 0, 0])

    save_image(polygon_img, '9')

    features |= score_lines(polygon_lines, long_lines)

    # clusters
    km = KMeans(n_init='auto', random_state=SEED)
    clustered = km.fit_predict(orto_img_denoise.reshape(-1, 1)).reshape(orto_img_denoise.shape)
    clustered_img = np.zeros_like(lines_edges_img)
    cluster_data = []

    for cluster in np.unique(clustered):
        cluster_mask = clustered == cluster
        clustered_img[cluster_mask] = random_color()

        mask_size = np.count_nonzero(cluster_mask)
        inside_mask_prop = np.count_nonzero(cluster_mask & polygon_mask)
        outside_mask_prop = np.count_nonzero(cluster_mask & non_polygon_mask)

        cluster_data.append({
            'size': mask_size / orto_img_denoise.size,
            'inside_rel': inside_mask_prop / mask_size,
            'inside_abs': inside_mask_prop / polygon_mask_count,
            'outside_rel': outside_mask_prop / mask_size,
            'outside_abs': outside_mask_prop / non_polygon_mask_count,
        })

    cluster_data.sort(key=lambda x: x['inside_abs'], reverse=True)

    for i, cluster in enumerate(cluster_data):
        for k, v in cluster.items():
            features[f'cluster_{i}_{k}'] = v

    save_image(clustered_img, '10')

    cluster_inside_abs = tuple(cluster['inside_abs'] for cluster in cluster_data)
    cluster_inside_abs_cumsum = np.cumsum(cluster_inside_abs)

    for n in (0.5, 0.6, 0.7, 0.8, 0.9):
        take_first = np.searchsorted(cluster_inside_abs_cumsum, n) + 1
        features[f'inside_clusters_{n}'] = take_first
        features[f'inside_clusters_{n}_acc'] = sum(d['inside_rel'] for d in cluster_data[:take_first]) / take_first

    return features
