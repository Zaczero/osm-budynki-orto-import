import random
from itertools import pairwise
from multiprocessing import Pool
from time import sleep

import numpy as np
from skimage import draw

from budynki import (Building, ClassifiedBuilding, building_chunks,
                     fetch_buildings)
from check_on_osm import check_on_osm
from config import CPU_COUNT, DRY_RUN, SEED, SLEEP_AFTER_ONE_IMPORT
from dataset import create_dataset, process_dataset
from db_added import filter_added, mark_added
from db_grid import iter_grid
from model import Model, create_model
from openstreetmap import OpenStreetMap
from orto import fetch_orto
from osm_change import create_buildings_change
from processor import process_polygon
from transform_px_geo import transform_geo_to_px
from utils import print_run_time, save_image

random.seed(SEED)
np.random.seed(SEED)


def _process_building(building: Building) -> tuple[Building, dict | None]:
    with print_run_time('Process building'):
        return building, process_polygon(building.polygon)


def main() -> None:
    with print_run_time('Loading model'):
        model = Model()

    with print_run_time('Logging in'):
        osm = OpenStreetMap()
        display_name = osm.get_authorized_user()['display_name']
        print(f'👤 Welcome, {display_name}!')

        changeset_max_size = osm.get_changeset_max_size()

    with Pool(CPU_COUNT) as pool:
        for cell in iter_grid():
            print(f'[CELL] ⚙️ Processing {cell!r}')

            with print_run_time('Fetch buildings'):
                buildings = fetch_buildings(cell)

            with print_run_time('Filter added buildings'):
                buildings = filter_added(buildings)

            if not buildings:
                print('[CELL] ⏭️ Nothing to do')
                continue

            valid_buildings: list[ClassifiedBuilding] = []

            if CPU_COUNT == 1:
                iterator = map(_process_building, buildings)
            else:
                iterator = pool.imap_unordered(_process_building, buildings)

            for building, data in iterator:
                if data is None:
                    print(f'[PROCESS] 🚫 Unsupported')
                    continue

                is_valid, proba = model.predict_single(data)

                if is_valid:
                    print(f'[PROCESS][{proba:.3f}] ✅ Valid')
                    valid_buildings.append(ClassifiedBuilding(building, proba))
                else:
                    print(f'[PROCESS][{proba:.3f}] 🚫 Invalid')
                    mark_added((building,), reason='predict', predict=proba)

            print(f'[CELL][1/2] 🏠 Valid buildings: {len(valid_buildings)}')

            with print_run_time('Check on OSM'):
                found, not_found = check_on_osm(valid_buildings)
                assert len(found) + len(not_found) == len(valid_buildings)

            if found:
                mark_added(tuple(map(lambda cb: cb.building, found)), reason='found')

            print(f'[CELL][2/2] 🏠 Needing import: {len(not_found)}')

            if not_found:
                for score_min, score_max, name in ((0.9, 1.1, '>90%',),
                                                   (0.0, 0.9, '>80%',)):
                    buildings = tuple(cb.building for cb in not_found if score_min <= cb.score < score_max)

                    for chunk in building_chunks(buildings, size=changeset_max_size):
                        with print_run_time('Create OSM change'):
                            osm_change = create_buildings_change(chunk)

                        with print_run_time('Upload OSM change'):
                            if DRY_RUN:
                                print('[DRY-RUN] 🚫 Skipping upload')
                            else:
                                osm.upload_osm_change(osm_change, name)

                        mark_added(chunk, reason='upload')
                        print(f'✅ Import successful: {name!r} ({len(chunk)})')

                if SLEEP_AFTER_ONE_IMPORT:
                    sleep_duration = len(not_found) * SLEEP_AFTER_ONE_IMPORT
                    print(f'[SLEEP-IMPORT] ⏳ Sleeping for {sleep_duration} seconds...')
                    sleep(sleep_duration)


if __name__ == '__main__':
    # create_dataset(3000)
    # process_dataset()
    # create_model()
    main()
