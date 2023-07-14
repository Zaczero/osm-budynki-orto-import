from typing import Sequence

import numpy as np

from box import Box
from latlon import LatLon


def transform_px_to_geo(lines: Sequence[tuple[tuple[float, float], tuple[float, float]]], box: Box, shape: tuple[int, int]) -> Sequence[tuple[LatLon, LatLon]]:
    p1 = box.point
    p2 = box.point + box.size

    x_res = (p2.lon - p1.lon) / shape[1]
    y_res = (p2.lat - p1.lat) / shape[0]

    lines_arr = np.array(lines)
    result = []

    for line in lines_arr:
        gy = p1.lat - (line[:, 1] + 0.5) * y_res
        gx = p1.lon + (line[:, 0] + 0.5) * x_res
        result.append((LatLon(gy[0], gx[0]), LatLon(gy[1], gx[1])))

    return tuple(result)


def transform_geo_to_px(lines: Sequence[tuple[LatLon, LatLon]], box: Box, shape: tuple[int, int]) -> Sequence[tuple[tuple[float, float], tuple[float, float]]]:
    p1 = box.point
    p2 = box.point + box.size

    x_res = (p2.lon - p1.lon) / shape[1]
    y_res = (p2.lat - p1.lat) / shape[0]

    lines_arr = np.array(lines)
    result = []

    for line in lines_arr:
        py = ((p2.lat - line[:, 0]) // y_res).astype(int)
        px = ((line[:, 1] - p1.lon) // x_res).astype(int)
        result.append(tuple(zip(py, px)))

    return tuple(result)
