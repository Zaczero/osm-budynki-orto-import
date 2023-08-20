from collections import defaultdict
from typing import Generator, Iterable, NamedTuple, Sequence

import httpx
import xmltodict
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from config import SKIP_CONSTRUCTION
from latlon import LatLon
from polygon import Polygon
from utils import http_headers


class Building(NamedTuple):
    polygon: Polygon
    tags: dict[str, str]


class ClassifiedBuilding(NamedTuple):
    building: Building
    score: float


@retry(wait=wait_exponential(), stop=stop_after_attempt(5), reraise=True)
def fetch_buildings(box: Box) -> Sequence[Building]:
    p1 = box.point
    p2 = box.point + box.size

    xmin = p1[1]
    xmax = p2[1]
    ymin = p1[0]
    ymax = p2[0]

    r = httpx.get('https://budynki.openstreetmap.org.pl/josm_data', params={
        'filter_by': 'bbox',
        'layers': 'buildings_to_import',
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
    }, headers=http_headers(), timeout=60)

    r.raise_for_status()

    data: dict = xmltodict.parse(r.text, force_list=('node', 'way', 'nd', 'tag'))['osm']

    node_id_used_by = defaultdict(set)
    node_id_to_latlon = {
        node['@id']: LatLon(float(node['@lat']), float(node['@lon']))
        for node in data.get('node', [])}

    result: dict[str, Building] = {}

    for way in data.get('way', []):
        way_id = way['@id']
        points = []

        for node in way.get('nd', []):
            node_id = node['@ref']
            node_id_used_by[node_id].add(way_id)
            points.append(node_id_to_latlon[node_id])

        assert len(points) > 3  # first and last point are the same

        building = Building(
            polygon=Polygon(tuple(points)),
            tags={
                tag['@k']: tag['@v']
                for tag in way.get('tag', [])})

        if SKIP_CONSTRUCTION and building.tags.get('building', None) == 'construction':
            continue

        result[way_id] = building

    # TODO: handle merged buildings, for now ignore them:
    # (osm change needs support too)
    for node_id, way_ids in node_id_used_by.items():
        if len(way_ids) == 1:
            continue

        print(f'[BUDYNKI] Node {node_id!r} is used by {len(way_ids)} ways, ignoring')

        for way_id in way_ids:
            result.pop(way_id, None)

    return tuple(result.values())


def building_chunks(buildings: Iterable[Building], *, size: int) -> Generator[Sequence[Building], None, None]:
    result = []
    result_size = 0

    for building in buildings:
        building_size = len(building.polygon.points) + 1  # +1 for way

        if result_size + building_size > size:
            yield result
            result.clear()
            result_size = 0

        result.append(building)
        result_size += building_size

    if result:
        yield result
