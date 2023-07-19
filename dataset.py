import json
from multiprocessing import Pool
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.io import imread

from budynki import Building, fetch_buildings
from config import CPU_COUNT, DATASET_DIR, IMAGES_DIR, MODEL_DATASET_PATH
from db_grid import random_grid
from latlon import LatLon
from model import Model
from polygon import Polygon
from processor import process_polygon
from utils import print_run_time, save_image


class DatasetEntry(NamedTuple):
    id: str
    label: bool
    data: dict | list
    image: np.ndarray

    def get_polygon(self) -> Polygon:
        return Polygon(tuple(LatLon(*p) for p in self.data[0]))


def _process_building(building: Building) -> tuple[Building, dict | None]:
    with print_run_time('Process building'):
        return building, process_polygon(building.polygon, create_dataset=True)


def create_dataset(size: int) -> None:
    with print_run_time('Loading model'):
        model = Model()

    with Pool(CPU_COUNT) as pool:
        for cell in random_grid():
            print(f'[CELL] ⚙️ Processing {cell!r}')

            with print_run_time('Fetch buildings'):
                buildings = fetch_buildings(cell)

            if not buildings:
                print('[CELL] ⏭️ Nothing to do')
                continue

            if CPU_COUNT == 1:
                iterator = map(_process_building, buildings)
            else:
                iterator = pool.imap_unordered(_process_building, buildings)

            for building, data in iterator:
                if data is None:
                    continue

                unique_id = building.polygon.unique_id_hash()
                data_extra = data['_']
                del data['_']

                _, proba = model.predict_single(data)

                if proba >= 0.5:
                    save_folder = '1'
                else:
                    save_folder = '0'

                save_image(data_extra['raw'], f'{save_folder}/{unique_id}', force=True)
                save_image(data_extra['overlay'], f'{save_folder}/{unique_id}_overlay1', force=True)
                save_image(data_extra['overlay_rgb'], f'{save_folder}/{unique_id}_overlay3', force=True)

                with open(IMAGES_DIR / f'{save_folder}/{unique_id}.json', 'w') as f:
                    json.dump(building.polygon, f)

                size -= 1

            if size <= 0:
                break


def _iter_dataset() -> Iterable[DatasetEntry]:
    for label in (1, 0):
        for p in sorted((DATASET_DIR / str(label)).glob('*.json')):
            identifier = p.stem
            print(f'[DATASET] 📄 Iterating: {identifier!r}')

            data = json.loads(p.read_text())
            image = imread(p.with_suffix('.png'))
            image = img_as_float(image)

            yield DatasetEntry(identifier, label, data, image)


def _process_dataset_entry(entry: DatasetEntry) -> dict | None:
    polygon = entry.get_polygon()
    process_result = process_polygon(polygon, orto_rgb_img=entry.image)

    if process_result is None:
        return None

    process_result['id'] = entry.id
    process_result['label'] = entry.label
    return process_result


def process_dataset() -> None:
    result = []

    with Pool(CPU_COUNT) as pool:
        if CPU_COUNT == 1:
            iterator = map(_process_dataset_entry, _iter_dataset())
        else:
            iterator = pool.imap_unordered(_process_dataset_entry, _iter_dataset())

        for process_result in iterator:
            if process_result is None:
                continue

            result.append(process_result)

    df = pd.DataFrame(result)
    df.to_csv(MODEL_DATASET_PATH, index=False)
