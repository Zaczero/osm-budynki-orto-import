from typing import Sequence

from shapely import geometry

from budynki import Building
from overpass import query_buildings


def check_on_osm(buildings: Sequence[Building]) -> tuple[Sequence[Building], Sequence[Building]]:
    building_polygons = query_buildings(tuple(b.polygon.get_bounding_box() for b in buildings))

    found = []
    not_found = []

    for building, polygons in zip(buildings, building_polygons):
        building_shape = geometry.Polygon(building.polygon.points)

        for polygon in polygons:
            polygon_shape = geometry.Polygon(polygon.points)

            if building_shape.intersects(polygon_shape):
                found.append(building)
                break
        else:
            not_found.append(building)

    return tuple(found), tuple(not_found)
