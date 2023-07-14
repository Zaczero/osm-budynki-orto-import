import random
from typing import Generator, Sequence

from box import Box
from config import DB_GRID
from latlon import LatLon

_GRID_SIZE = 0.25
_COUNTRY_BB = Box(LatLon(49.0, 14.0),
                  LatLon(55.0, 24.25) - LatLon(49.0, 14.0))


def _get_last_index() -> int:
    doc = DB_GRID.get(doc_id=1)

    if doc is None:
        return -1

    return doc['index']


def _set_last_index(index: int) -> None:
    DB_GRID.upsert({'index': index}, lambda _: True)


def iter_grid() -> Generator[Box, None, None]:
    while True:
        last_index = _get_last_index()

        if last_index > -1:
            print(f'[GRID] ⏭️ Resuming from index {last_index + 1}')

        index = 0

        for y in range(int(_COUNTRY_BB.size.lat / _GRID_SIZE)):
            for x in range(int(_COUNTRY_BB.size.lon / _GRID_SIZE)):
                if index > last_index:
                    yield Box(
                        point=_COUNTRY_BB.point + LatLon(y * _GRID_SIZE, x * _GRID_SIZE),
                        size=LatLon(_GRID_SIZE, _GRID_SIZE))

                    _set_last_index(index)
                    print(f'[GRID] ☑️ Finished index {index}')

                index += 1

        _set_last_index(-1)
