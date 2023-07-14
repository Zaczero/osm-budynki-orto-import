from itertools import combinations
from math import radians
from typing import Iterable, Sequence

import networkx as nx
from rtree import index
from shapely.geometry import LineString, Point

from line import Line


def _find_farthest_points(lines: Iterable[LineString]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = (Point(coord) for line in lines for coord in line.coords)
    p0, p1 = max(combinations(points, 2), key=lambda pair: pair[0].distance(pair[1]))
    return (p0.x, p0.y), (p1.x, p1.y)


def merge_lines(lines: Sequence[tuple[tuple[float, float], tuple[float, float]]], *, dist_threshold: float, angle_threshold: float) -> Sequence[tuple[tuple[float, float], tuple[float, float]]]:
    angle_threshold = radians(angle_threshold)
    line_objs = [Line(line, expand_bbox=dist_threshold) for line in lines]

    idx = index.Index()

    for i, line in enumerate(line_objs):
        idx.insert(i, line.bbox)

    # node: line
    # edge: valid merge
    G = nx.Graph()

    for i, line1 in enumerate(line_objs):
        nearby_lines = idx.intersection(line1.bbox)

        for j in nearby_lines:
            if i == j:
                continue

            line2 = line_objs[j]

            angle_diff = abs(line1.angle - line2.angle)
            if angle_diff > angle_threshold:
                continue

            dist = line1.line.distance(line2.line)
            if dist > dist_threshold:
                continue

            G.add_edge(i, j)

    result = []

    for component in nx.connected_components(G):
        lines_to_merge = (line_objs[i].line for i in component)
        p1, p2 = _find_farthest_points(lines_to_merge)
        result.append((p1, p2))

    for i, line in enumerate(line_objs):
        if i not in G:
            result.append(line.line.coords)

    return tuple((
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])))
        for p1, p2 in result)
