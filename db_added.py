from time import time
from typing import Iterable, Sequence

from pymongo import DESCENDING, InsertOne

from budynki import Building
from config import MONGO_ADDED, SCORER_VERSION, VERSION

# def migrate():
#     for doc in DB_ADDED.all():
#         doc = dict(doc)
#         MONGO_ADDED.insert_one(doc)


def filter_added(buildings: Iterable[Building]) -> Sequence[Building]:
    def _is_added(building: Building) -> bool:
        unique_id = building.polygon.unique_id_hash(8)

        doc = MONGO_ADDED.find({
            'unique_id': unique_id,
        }).sort({
            'scorer_version': DESCENDING,
            'timestamp': DESCENDING,
        }).limit(1)

        if doc is None:
            return False

        return (
            doc['scorer_version'] >= SCORER_VERSION or
            doc['reason'] in {'found', 'upload'}
        )

    return tuple(filter(lambda b: not _is_added(b), buildings))


def mark_added(buildings: Sequence[Building], **kwargs) -> None:
    MONGO_ADDED.bulk_write([
        InsertOne({
            'timestamp': time(),
            'unique_id': building.polygon.unique_id_hash(8),
            'location': tuple(building.polygon.points[0]),
            'app_version': VERSION,
            'scorer_version': SCORER_VERSION,
            **kwargs,
        }) for building in buildings
    ], ordered=False)
