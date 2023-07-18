import json
from multiprocessing import Pool
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.io import imread

from config import CPU_COUNT, DATASET_DIR
from latlon import LatLon
from polygon import Polygon
from processor import process_polygon


class DatasetEntry(NamedTuple):
    id: str
    label: bool
    data: dict
    image: np.ndarray

    def get_polygon(self) -> Polygon:
        return Polygon(tuple(LatLon(*p) for p in self.data['polygon']))


def _iter_dataset() -> Iterable[DatasetEntry]:
    for label in (False, True):
        for p in sorted((DATASET_DIR / str(label)).glob('*.json')):
            identifier = p.stem
            print(f'[DATASET] 📄 Iterating: {identifier!r}')

            data = json.loads(p.read_text())
            image = imread(p.with_suffix('.png'))
            image = img_as_float(image)

            yield DatasetEntry(identifier, label, data, image)


def _process_dataset_entry(entry: DatasetEntry) -> dict | None:
    polygon = entry.get_polygon()
    process_result = process_polygon(polygon, orto_img=entry.image)

    if process_result is None:
        return None

    process_result['id'] = entry.id
    process_result['label'] = entry.label
    return process_result


def process_dataset() -> None:
    result = []

    with Pool(CPU_COUNT) as pool:
        for process_result in pool.imap_unordered(_process_dataset_entry, _iter_dataset()):
            if process_result is None:
                continue

            result.append(process_result)

    df = pd.DataFrame(result)
    df.to_csv(DATASET_DIR / 'dataset.csv', index=False)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / 'dataset.csv')
