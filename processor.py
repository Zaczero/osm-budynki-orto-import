from itertools import chain, pairwise
from operator import itemgetter
from typing import Sequence

import numpy as np
from scipy.stats import kurtosis, skew
from shapely import geometry
from shapely.geometry import LineString
from skimage import (color, draw, exposure, feature, filters, img_as_float,
                     img_as_ubyte, morphology, restoration, transform)
from skimage.color.adapt_rgb import adapt_rgb, each_channel
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


@adapt_rgb(each_channel)
def median_rgb(image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    return filters.median(image, selem)


def safe_var(x) -> float:
    return np.var(x) if x.size > 1 else 0


def process_polygon(polygon: Polygon, orto_rgb_img: np.ndarray | None = None, *, create_dataset: bool = False) -> dict | None:
    features = {}
    polygon_box = polygon.get_bounding_box()
    orto_box = polygon_box.extend(meters=FETCH_EXTEND)

    if orto_rgb_img is None:
        orto_rgb_img = img_as_float(fetch_orto(orto_box))

    assert orto_rgb_img.dtype == float
    save_image(orto_rgb_img, '1')

    if create_dataset:
        features['_'] = {}
        features['_']['raw'] = orto_rgb_img

    polygon_shape = geometry.Polygon(polygon.points)
    features['area'] = polygon_shape.area
    features['perimeter'] = polygon_shape.length
    features['perimeter_compactness'] = features['perimeter'] ** 2 / (4 * np.pi * features['area'])
    minx, miny, maxx, maxy = polygon_shape.bounds
    aspect_ratio = (maxx - minx) / (maxy - miny) if (maxy - miny) != 0 else 0
    features['aspect_ratio'] = aspect_ratio
    features['convexity'] = polygon_shape.area / polygon_shape.convex_hull.area

    polygon_lines = transform_geo_to_px(tuple(e for e in pairwise(polygon.points)), orto_box, orto_rgb_img.shape[:2])
    polygon_mask = get_polygon_mask(polygon_lines, orto_rgb_img.shape[:2])
    non_polygon_mask = ~polygon_mask

    polygon_mask = morphology.erosion(polygon_mask, morphology.disk(3))
    surrounding_mask = morphology.dilation(polygon_mask, morphology.disk(15)) & non_polygon_mask
    non_polygon_mask = morphology.erosion(non_polygon_mask, morphology.disk(5))

    polygon_mask_count = np.count_nonzero(polygon_mask)
    surrounding_mask_count = np.count_nonzero(surrounding_mask)
    non_polygon_mask_count = np.count_nonzero(non_polygon_mask)

    no_mask = np.ones_like(polygon_mask)

    if not polygon_mask_count or not non_polygon_mask_count:
        print('[PROCESS] Polygon mask is empty')
        return None

    # normalize saturation and value
    orto_hsv_img = color.rgb2hsv(orto_rgb_img)
    s_channel = orto_hsv_img[:, :, 1]
    p0 = np.min(s_channel)
    p100 = np.max(s_channel)
    s_channel = exposure.rescale_intensity(s_channel, in_range=(p0, p100))
    orto_hsv_img[:, :, 1] = s_channel
    v_channel = orto_hsv_img[:, :, 2]
    p05 = np.percentile(v_channel, 5)
    p95 = np.percentile(v_channel, 95)
    v_channel = exposure.rescale_intensity(v_channel, in_range=(p05, p95))
    orto_hsv_img[:, :, 2] = v_channel
    orto_rgb_img = color.hsv2rgb(orto_hsv_img)
    save_image(orto_rgb_img, '2')

    orto_img = color.rgb2gray(orto_rgb_img)
    save_image(orto_img, '3')

    # denoise
    orto_img_denoise: np.ndarray = restoration.denoise_nl_means(orto_img)
    save_image(orto_img_denoise, '4')

    orto_rgb_img_denoise: np.ndarray = median_rgb(orto_rgb_img, morphology.disk(3))
    save_image(orto_rgb_img_denoise, '5')

    if create_dataset:
        overlay_img = img_as_float(color.gray2rgb(orto_img))
        overlay_rgb_img = orto_rgb_img.copy()

        for line_points in polygon_lines:
            rr, cc = draw.line(*line_points[0], *line_points[1])
            overlay_img[rr, cc] = overlay_rgb_img[rr, cc] = np.array([1, 0, 0])

        features['_']['overlay'] = overlay_img
        features['_']['overlay_rgb'] = overlay_rgb_img

    for img, img_prefix in ((orto_img, ''), (orto_rgb_img[:, :, 0], 'R_'), (orto_rgb_img[:, :, 1], 'G_'), (orto_rgb_img[:, :, 2], 'B_'), (orto_img_denoise, 'denoise_')):
        for mask, mask_prefix in ((no_mask, ''), (polygon_mask, 'inside_'), (surrounding_mask, 'around_'), (non_polygon_mask, 'outside_')):
            prefix = img_prefix + mask_prefix

            # 2D features
            masked = img_as_ubyte(img)
            masked[~mask] = 0
            glcm = graycomatrix(masked, (1,), (0, np.pi/4, np.pi/2, 3*np.pi/4), levels=256, symmetric=True, normed=True)

            for prop in ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'):
                features[f'{prefix}{prop}'] = graycoprops(glcm, prop)[0, 0]

            # histogram of gradients
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

            # sharpness
            laplacian: np.ndarray = filters.laplace(masked)
            features[f'{prefix}sharpness'] = laplacian[mask].var()

            # fractal dimension
            scales = np.arange(7, 0, -1)
            ns = []
            for scale in scales:
                window = max(1, int(2 ** scale))
                blocks = np.lib.stride_tricks.sliding_window_view(masked, (window, window))
                ns.append(np.count_nonzero(blocks, axis=(1, 3)).mean())
            coeffs = np.polyfit(np.log2(ns), scales, 1)
            features[f'{prefix}fractal_dimension'] = -coeffs[0]

            # local binary patterns
            lbp = feature.local_binary_pattern(masked, 8, 1)
            lbp_hist, _ = np.histogram(lbp, density=True, bins=8, range=(0, 8))
            for i, val in enumerate(lbp_hist):
                features[f'{prefix}lbp_{i}'] = val

            # gabor
            gabor_real, gabor_imag = filters.gabor(masked, frequency=0.6)
            features[f'{prefix}gabor_real_mean'] = np.mean(gabor_real)
            features[f'{prefix}gabor_real_var'] = np.var(gabor_real)
            features[f'{prefix}gabor_imag_mean'] = np.mean(gabor_imag)
            features[f'{prefix}gabor_imag_var'] = np.var(gabor_imag)

            # entropy
            features[f'{prefix}entropy'] = shannon_entropy(masked)

            # 1D features
            masked = img[mask]
            features[f'{prefix}mean'] = masked.mean()
            features[f'{prefix}var'] = masked.var()
            features[f'{prefix}skew'] = skew(masked)
            features[f'{prefix}kurtosis'] = kurtosis(masked)

    # shadow detection
    thresh: np.ndarray = filters.threshold_otsu(orto_img_denoise)
    shadow = orto_img_denoise < thresh
    features['inside_shadow'] = np.count_nonzero(shadow & polygon_mask) / polygon_mask_count
    features['around_shadow'] = np.count_nonzero(shadow & surrounding_mask) / non_polygon_mask_count
    features['outside_shadow'] = np.count_nonzero(shadow & non_polygon_mask) / non_polygon_mask_count

    features['shadow_ratio'] = features['inside_shadow'] / features['outside_shadow']

    # edge detection
    edges = feature.canny(orto_img_denoise)
    # edges = morphology.dilation(edges, morphology.disk(3))
    # edges = morphology.erosion(edges, morphology.disk(3))
    edges = morphology.dilation(edges, morphology.disk(1))
    save_image(edges, '6')

    # measure edge noise
    features['inside_noise_var'] = safe_var(orto_img_denoise[edges & polygon_mask])
    features['inside_noise_perc'] = np.count_nonzero(edges & polygon_mask) / polygon_mask_count
    features['around_noise_var'] = safe_var(orto_img_denoise[edges & surrounding_mask])
    features['around_noise_perc'] = np.count_nonzero(edges & surrounding_mask) / non_polygon_mask_count
    features['outside_noise_var'] = safe_var(orto_img_denoise[edges & non_polygon_mask])
    features['outside_noise_perc'] = np.count_nonzero(edges & non_polygon_mask) / non_polygon_mask_count

    features['noise_ratio'] = features['inside_noise_perc'] / features['outside_noise_perc']

    lines = chain(
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 1),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 2),
        transform.probabilistic_hough_line(edges, line_length=15, line_gap=1, rng=SEED + 3))
    lines = tuple(((p0[1], p0[0]), (p1[1], p1[0])) for p0, p1 in lines)  # fix x, y order
    lines_edges_img = img_as_float(color.gray2rgb(edges))
    lines_img = np.zeros_like(lines_edges_img)

    for line_points in lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        lines_edges_img[rr, cc] = lines_img[rr, cc] = random_color()

    save_image(lines_edges_img, '7')
    save_image(lines_img, '8')

    merged_lines = merge_lines(lines, dist_threshold=2, angle_threshold=10)
    merged_lines_img = np.zeros_like(lines_edges_img)

    for line_points in merged_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        merged_lines_img[rr, cc] = random_color()

    save_image(merged_lines_img, '9')

    long_lines = tuple(line for line in merged_lines if LineString(line).length > 20)
    long_lines_img = np.zeros_like(lines_edges_img)

    for line_points in long_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        long_lines_img[rr, cc] = random_color()

    save_image(long_lines_img, '10')

    polygon_img = np.copy(long_lines_img)

    for line_points in polygon_lines:
        rr, cc = draw.line(*line_points[0], *line_points[1])
        polygon_img[rr, cc] = np.array([1, 0, 0])

    save_image(polygon_img, '11')

    features |= score_lines(polygon_lines, long_lines)

    # clusters
    km = KMeans(n_init='auto', random_state=SEED)
    clustered = km.fit_predict(orto_rgb_img_denoise.reshape(-1, 3)).reshape(orto_img_denoise.shape)
    clustered_img = np.zeros_like(lines_edges_img)
    cluster_data = []

    for cluster in np.unique(clustered):
        cluster_mask = clustered == cluster
        clustered_img[cluster_mask] = random_color()

        cluster_size = np.count_nonzero(cluster_mask)
        inside_cluster = np.count_nonzero(cluster_mask & polygon_mask)
        surround_cluster = np.count_nonzero(cluster_mask & surrounding_mask)
        outside_cluster = np.count_nonzero(cluster_mask & non_polygon_mask)

        cluster_data.append({
            'proportion': cluster_size / orto_img_denoise.size,
            'inside_rel': inside_cluster / cluster_size,
            'inside_abs': inside_cluster / polygon_mask_count,
            'around_rel': surround_cluster / cluster_size,
            'around_abs': surround_cluster / surrounding_mask_count,
            'outside_rel': outside_cluster / cluster_size,
            'outside_abs': outside_cluster / non_polygon_mask_count,
        })

    cluster_data.sort(key=lambda x: x['proportion'], reverse=True)  # largest first

    for i, cluster in enumerate(cluster_data):
        for k, v in cluster.items():
            features[f'cluster_{i}_{k}'] = v

    save_image(clustered_img, '12')

    cluster_data.sort(key=lambda x: x['inside_abs'], reverse=True)  # largest first
    cluster_inside_abs = tuple(cluster['inside_abs'] for cluster in cluster_data)
    cluster_inside_abs_cumsum = np.cumsum(cluster_inside_abs)

    for n in (0.5, 0.6, 0.7, 0.8, 0.9):
        take_first = np.searchsorted(cluster_inside_abs_cumsum, n) + 1
        features[f'inside_clusters_{n}'] = take_first
        features[f'inside_clusters_{n}_acc'] = sum(d['inside_rel'] for d in cluster_data[:take_first]) / take_first

    return features
