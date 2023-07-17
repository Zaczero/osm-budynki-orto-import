from typing import NamedTuple, Self

from latlon import LatLon
from utils import meters_to_lat, meters_to_lon


class Box(NamedTuple):
    point: LatLon
    size: LatLon

    def extend(self, meters: float) -> Self:
        extend_lat = meters_to_lat(meters)
        extend_lon = meters_to_lon(meters, self.point.lat)

        return Box(
            point=LatLon(self.point.lat - extend_lat, self.point.lon - extend_lon),
            size=LatLon(self.size.lat + extend_lat * 2, self.size.lon + extend_lon * 2))
