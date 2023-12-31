import random
import traceback
from multiprocessing import Pool
from time import sleep

import numpy as np
from httpx import HTTPStatusError

from budynki import (Building, ClassifiedBuilding, building_chunks,
                     fetch_buildings)
from check_on_osm import check_on_osm
from config import (CONFIDENCE, CPU_COUNT, DRY_RUN, PROCESS_NICE, SEED,
                    SLEEP_AFTER_ONE_IMPORT)
from dataset import create_dataset
from db_added import filter_added, mark_added
from db_grid import iter_grid
from model import create_model
from openstreetmap import OpenStreetMap
from osm_change import create_buildings_change
from processor import process_image, process_polygon
from tuned_model import TunedModel
from utils import print_run_time, set_nice

random.seed(SEED)
np.random.seed(SEED)


def _process_building(building: Building) -> tuple[Building, np.ndarray | None]:
    with print_run_time('Process building'):
        try:
            polygon_result = process_polygon(building.polygon)
            return building, process_image(polygon_result.image, polygon_result.mask)
        except:
            traceback.print_exc()
            return building, None


def main() -> None:
    set_nice(PROCESS_NICE)

    with print_run_time('Logging in'):
        osm = OpenStreetMap()
        display_name = osm.get_authorized_user()['display_name']
        print(f'👤 Welcome, {display_name}!')

        changeset_max_size = osm.get_changeset_max_size()

    with print_run_time('Loading model'):
        model = TunedModel()

    with Pool(CPU_COUNT) as pool:
        for cell in iter_grid():
            print(f'[CELL] ⚙️ Processing {cell!r}')

            try:
                with print_run_time('Fetch buildings'):
                    buildings = fetch_buildings(cell)
            except HTTPStatusError as e:
                if e.response.status_code == 500:
                    print('[CELL] 🚫 Corrupted data, skipping')
                    continue

                raise

            with print_run_time('Filter added buildings'):
                buildings = filter_added(buildings)

            if not buildings:
                print('[CELL] ⏭️ Nothing to do')
                continue

            valid_buildings: list[ClassifiedBuilding] = []

            if CPU_COUNT == 1:
                iterator = map(_process_building, buildings)
            else:
                iterator = pool.imap_unordered(_process_building, buildings, chunksize=4)

            for building, model_input in iterator:
                if model_input is None:
                    print(f'[PROCESS] 🚫 Unsupported')
                    mark_added((building,), reason='unsupported')
                    continue

                is_valid, proba = model.predict_single(model_input)

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
                not_found_sorted = sorted(not_found, key=lambda cb: cb.score)
                confidence_threshold = CONFIDENCE + (1 - CONFIDENCE) / 2
                confidence_index = next(
                    (i for i, cb in enumerate(not_found_sorted) if cb.score > confidence_threshold),
                    len(not_found_sorted))
                not_found_low = not_found_sorted[:confidence_index]
                not_found_high = not_found_sorted[confidence_index:]

                for collection, name in ((not_found_high, f'b.wysoka pewność'),
                                         (not_found_low, f'wysoka pewność')):
                    buildings = tuple(cb.building for cb in collection)

                    for chunk in building_chunks(buildings, size=changeset_max_size):
                        with print_run_time('Create OSM change'):
                            osm_change = create_buildings_change(chunk)

                        with print_run_time('Upload OSM change'):
                            if DRY_RUN:
                                print('[DRY-RUN] 🚫 Skipping upload')
                                changeset_id = 0
                            else:
                                changeset_id = osm.upload_osm_change(osm_change, name)

                        mark_added(chunk, reason='upload', changeset_id=changeset_id)
                        print(f'✅ Import successful: {name!r} ({len(chunk)})')

                if SLEEP_AFTER_ONE_IMPORT:
                    sleep_duration = len(not_found) * SLEEP_AFTER_ONE_IMPORT
                    print(f'[SLEEP-IMPORT] ⏳ Sleeping for {sleep_duration} seconds...')
                    sleep(sleep_duration)


if __name__ == '__main__':
    # create_dataset(5)
    # process_dataset()
    # create_model()
    main()
