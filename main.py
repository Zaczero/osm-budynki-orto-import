import random
from multiprocessing import Pool
from time import sleep

import numpy as np

from budynki import Building, building_chunks, fetch_buildings
from check_on_osm import check_on_osm
from config import CPU_COUNT, DRY_RUN, SEED, SLEEP_AFTER_IMPORT
from db_added import filter_added, mark_added
from db_grid import iter_grid
from openstreetmap import OpenStreetMap
from osm_change import create_buildings_change
from processor import process_polygon
from utils import print_run_time

random.seed(SEED)
np.random.seed(SEED)


SCORE_THRESHOLD = 0.6


def _process_building(building: Building) -> tuple[Building, float]:
    with print_run_time('Process building'):
        return building, process_polygon(building.polygon)


def main() -> None:
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

            valid_buildings = []

            if CPU_COUNT == 1:
                iterator = map(_process_building, buildings)
            else:
                iterator = pool.imap_unordered(_process_building, buildings)

            for building, score in iterator:
                if score > SCORE_THRESHOLD:
                    print(f'[PROCESS][{score:05.3f}] 🟢 Valid')
                    valid_buildings.append(building)
                else:
                    print(f'[PROCESS][{score:05.3f}] 🔴 Invalid')
                    mark_added((building,), reason='score', score=score)

            print(f'[CELL][1/2] 🏠 Valid buildings: {len(valid_buildings)}')

            with print_run_time('Check on OSM'):
                found, not_found = check_on_osm(valid_buildings)
                assert len(found) + len(not_found) == len(valid_buildings)

            if found:
                mark_added(found, reason='found')

            print(f'[CELL][2/2] 🏠 Needing import: {len(not_found)}')

            if not_found:
                for chunk in building_chunks(not_found, size=changeset_max_size):
                    with print_run_time('Create OSM change'):
                        osm_change = create_buildings_change(chunk)

                    with print_run_time('Upload OSM change'):
                        if DRY_RUN:
                            print('[DRY-RUN] 🚫 Skipping upload')
                        else:
                            osm.upload_osm_change(osm_change)

                    mark_added(chunk, reason='upload')
                    print('✅ Import successful')

                if SLEEP_AFTER_IMPORT:
                    print(f'[SLEEP] ⏳ Sleeping for {SLEEP_AFTER_IMPORT} seconds...')
                    sleep(SLEEP_AFTER_IMPORT)


if __name__ == '__main__':
    main()
