from datetime import datetime, timedelta
from typing import Iterable, Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from config import OVERPASS_API_INTERPRETER
from latlon import LatLon
from polygon import Polygon
from utils import http_headers


def _split_by_count(elements: Iterable[dict]) -> list[list[dict]]:
    result = []
    current_split = []

    for e in elements:
        if e['type'] == 'count':
            result.append(current_split)
            current_split = []
        else:
            current_split.append(e)

    assert not current_split, 'Last element must be count type'
    return result


def _build_buildings_query(boxes: Iterable[Box], timeout: int) -> str:
    buffer = 0.00001
    return (
        f'[out:json][timeout:{timeout}];' +
        f''.join(
            f'way["building"]({box.point.lat-buffer},{box.point.lon-buffer},{box.point.lat+box.size.lat+buffer},{box.point.lon+box.size.lon+buffer});'
            f'out body qt;'
            f'>;'
            f'out skel qt;'
            f'out count;'
            for box in boxes)
    )


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def query_buildings(boxes: Sequence[Box]) -> Sequence[Sequence[Polygon]]:
    result = tuple([] for _ in boxes)

    for years_ago in (None, 1, 2):
        timeout = 180
        query = _build_buildings_query(boxes, timeout)

        if years_ago:
            date = datetime.utcnow() - timedelta(days=365 * years_ago)
            date_fmt = date.strftime('%Y-%m-%dT%H:%M:%SZ')
            query = f'[date:"{date_fmt}"]{query}'

        r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
        r.raise_for_status()

        elements = r.json()['elements']

        for i, slice in enumerate(_split_by_count(elements)):
            ways = (e for e in slice if e['type'] == 'way')
            node_id_to_latlon = {
                node['id']: LatLon(float(node['lat']), float(node['lon']))
                for node in slice if node['type'] == 'node'}

            for way in ways:
                points = tuple(node_id_to_latlon[node_id] for node_id in way['nodes'])
                result[i].append(Polygon(points))

    return result
