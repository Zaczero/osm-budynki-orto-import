from typing import NamedTuple, Self

from latlon import LatLon
from utils import meters_to_lat


class Box(NamedTuple):
    point: LatLon
    size: LatLon

    def extend(self, meters: float) -> Self:
        diff = meters_to_lat(meters)

        return Box(
            point=LatLon(self.point.lat - diff, self.point.lon - diff),
            size=LatLon(self.size.lat + diff * 2, self.size.lon + diff * 2))
