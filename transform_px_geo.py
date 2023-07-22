from typing import Sequence

from box import Box
from polygon import Polygon


def transform_rad_to_px(polygon: Polygon, box: Box, shape: tuple[int, int]) -> Sequence[tuple[int, int]]:
    y_res = box.size.lat / shape[0]
    x_res = box.size.lon / shape[1]

    result = []

    for point in polygon.points:
        y = int((box.point.lat + box.size.lat - point.lat) / y_res)
        x = int((point.lon - box.point.lon) / x_res)
        result.append((y, x))

    return tuple(result)
