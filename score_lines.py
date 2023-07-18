from math import radians, sqrt
from operator import itemgetter
from typing import NamedTuple, Self, Sequence

import numpy as np
from rtree import index
from shapely.geometry import LineString, Point

from line import Line

_DISTANCE_THRESHOLD = 15
_DISTANCE_SCORE_MIN = 8
_DISTANCE_SCORE_MAX = 12

_ANGLE_THRESHOLD = radians(20)
_ANGLE_SCORE_MIN = radians(15)
_ANGLE_SCORE_MAX = radians(20)


class PointSummary(NamedTuple):
    score: float
    distance_score: float
    angle_score: float
    distance: float
    angle: float

    @classmethod
    def zero(cls) -> Self:
        return cls(0, 0, 0, 100, 100)


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


def score_lines(edges: Sequence[tuple[tuple[float, float], tuple[float, float]]], lines: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> dict:
    line_objs = tuple(Line(line, expand_bbox=_DISTANCE_THRESHOLD) for line in lines)

    idx = index.Index()

    for i, line in enumerate(line_objs):
        idx.insert(i, line.bbox)

    edge_angles = []
    total_point_summary = []

    for i, edge_ in enumerate(edges):
        edge = Line(edge_, expand_bbox=_DISTANCE_THRESHOLD)
        edge_angles.append(edge.angle)

        points_along_edge = tuple(
            edge.line.interpolate(d, normalized=True).coords[0]
            for d in np.linspace(0, 1, num=int(edge.line.length / 2)))

        if not points_along_edge:
            continue

        dx, dy = _perpendicular_direction(*edge.line.coords)
        dx *= 10
        dy *= 10

        point_summary = [PointSummary.zero()] * len(points_along_edge)

        for j in idx.intersection(edge.bbox):
            line = line_objs[j]

            angle_diff = abs(line.angle - edge.angle)
            if angle_diff > _ANGLE_THRESHOLD:
                continue

            line_distance = edge.line.distance(line.line)
            if line_distance > _DISTANCE_THRESHOLD:
                continue

            for k, p in enumerate(points_along_edge):
                perpendicular_line = LineString((
                    (p[0] - dy, p[1] - dx),
                    (p[0] + dy, p[1] + dx)))

                if perpendicular_line.intersects(line.line):  # this is necessary to avoid warning
                    point_intersection = perpendicular_line.intersection(line.line)
                    point_distance = Point(p[0], p[1]).distance(point_intersection)

                    distance_score = _score(point_distance, _DISTANCE_SCORE_MIN, _DISTANCE_SCORE_MAX)
                    angle_score = _score(angle_diff, _ANGLE_SCORE_MIN, _ANGLE_SCORE_MAX)
                    score = distance_score * angle_score

                    if point_summary[k].score < score:
                        point_summary[k] = PointSummary(score, distance_score, angle_score, point_distance, angle_diff)

        total_point_summary.extend(point_summary)

    if not total_point_summary:
        total_point_summary = [PointSummary.zero()]

    result = {
        'point_rate': sum(1 for ps in total_point_summary if ps.score > 0) / len(total_point_summary),
    }

    for field in PointSummary._fields:
        values = np.array(tuple(map(itemgetter(PointSummary._fields.index(field)), total_point_summary)))

        result[f'point_{field}_mean'] = values.mean()
        result[f'point_{field}_std'] = values.std()
        result[f'point_{field}_var'] = values.var()

        for p in range(10, 90 + 1, 10):
            result[f'point_{field}_{p}'] = np.percentile(values, p)

    return result
