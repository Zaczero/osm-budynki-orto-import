from math import radians, sqrt
from statistics import mean
from typing import Sequence

import numpy as np
from rtree import index
from shapely.geometry import LineString, Point

from line import Line

# line distance
_DISTANCE_THRESHOLD = 10

# point distance
_DISTANCE_MIN = 7
_DISTANCE_MAX = 11

_ANGLE_MIN = radians(15)
_ANGLE_MAX = radians(20)


def _perpendicular_direction(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    dx, dy = p2[1] - p1[1], p2[0] - p1[0]
    length = sqrt(dx ** 2 + dy ** 2)
    return dy / length, -dx / length


def _score(x: float, min: float, max: float) -> float:
    if x < min:
        return 1
    elif x > max:
        return 0
    else:
        return (max - x) / (max - min)


def _length(line: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2)


def score_lines(edges: Sequence[tuple[tuple[float, float], tuple[float, float]]], lines: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    line_objs = tuple(Line(line, expand_bbox=_DISTANCE_THRESHOLD) for line in lines)

    idx = index.Index()

    for i, line in enumerate(line_objs):
        idx.insert(i, line.bbox)

    normalized_score = 0
    edge_lengths = tuple(_length(edge) for edge in edges)
    sum_edge_lengths = sum(edge_lengths)

    for i, (edge_, edge_length) in enumerate(zip(edges, edge_lengths)):
        edge = Line(edge_, expand_bbox=_DISTANCE_THRESHOLD)

        points_along_edge = tuple(
            edge.line.interpolate(d, normalized=True).coords[0]
            for d in np.linspace(0, 1, num=int(edge.line.length // 2)))

        if not points_along_edge:
            continue

        dx, dy = _perpendicular_direction(*edge.line.coords)
        dx *= 10
        dy *= 10

        point_scores = [0] * len(points_along_edge)

        for j in idx.intersection(edge.bbox):
            line = line_objs[j]

            if edge.line.distance(line.line) > _DISTANCE_THRESHOLD:
                continue

            angle_diff = line.angle - edge.angle
            angle_score = _score(abs(angle_diff), _ANGLE_MIN, _ANGLE_MAX)
            if angle_score == 0:
                continue

            for k, p in enumerate(points_along_edge):
                perpendicular_line = LineString((
                    (p[0] - dy, p[1] - dx),
                    (p[0] + dy, p[1] + dx)))

                if perpendicular_line.intersects(line.line):  # this is necessary to avoid warning
                    point_intersection = perpendicular_line.intersection(line.line)

                    inter_distance = Point(p[0], p[1]).distance(point_intersection)
                    inter_distance_score = _score(inter_distance, _DISTANCE_MIN, _DISTANCE_MAX)

                    point_score = angle_score * inter_distance_score
                    point_scores[k] = max(point_scores[k], point_score)

        normalized_score += mean(point_scores) * edge_length / sum_edge_lengths

    return normalized_score
