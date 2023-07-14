from time import time
from typing import Iterable, Sequence

from tinydb import Query

from budynki import Building
from config import DB_ADDED, SCORER_VERSION, VERSION


def _is_added(building: Building) -> bool:
    entry = Query()
    return DB_ADDED.contains(
        (entry.unique_id == building.polygon.unique_id()) &
        (entry.scorer_version >= SCORER_VERSION))


def filter_added(buildings: Iterable[Building]) -> Sequence[Building]:
    return tuple(filter(lambda poi: not _is_added(poi), buildings))


def mark_added(buildings: Iterable[Building], **kwargs) -> Sequence[int]:
    return DB_ADDED.insert_multiple({
        'timestamp': time(),
        'unique_id': b.polygon.unique_id(),
        'app_version': VERSION,
        'scorer_version': SCORER_VERSION,
    } | kwargs for b in buildings)
