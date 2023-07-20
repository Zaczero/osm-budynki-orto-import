import json
from multiprocessing import Pool
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import xmltodict
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

    cvat_annotations = {
        'annotations': {
            'version': '1.1',
            'image': []
        }
    }

    cvat_image_annotations: list[dict] = cvat_annotations['annotations']['image']

    with Pool(CPU_COUNT) as pool:
        for cell in random_grid():
            print(f'[CELL] ⚙️ Processing {cell!r}')

            with print_run_time('Fetch buildings'):
                buildings = fetch_buildings(cell)

            if not buildings:
                print('[CELL] ⏭️ Nothing to do')
                continue

            buildings = buildings[:(size - len(cvat_image_annotations))]

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

                is_valid, proba = model.predict_single(data, threshold=0.5)
                label = 'good' if is_valid else 'bad'

                raw_path = save_image(data_extra['raw'], f'CVAT/images/{unique_id}', force=True)
                raw_name = raw_path.name
                raw_name_safe = raw_name.replace('.', '_')

                save_image(data_extra['overlay_rgb'], f'CVAT/images/related_images/{raw_name_safe}/overlay', force=True)

                with open(IMAGES_DIR / f'CVAT/images/related_images/{raw_name_safe}/polygon.json', 'w') as f:
                    json.dump(building.polygon, f)

                annotation = {
                    '@name': f'images/{raw_name}',
                    '@width': data_extra['raw'].shape[1],
                    '@height': data_extra['raw'].shape[0],
                    'tag': [{
                        '@label': label,
                        '@source': 'auto',
                        'attribute': [{
                            '@name': 'proba',
                            '#text': f'{proba:.3f}'
                        }]
                    }]
                }

                cvat_image_annotations.append(annotation)

            if len(cvat_image_annotations) >= size:
                break

    # sort in lexical order
    cvat_image_annotations.sort(key=lambda x: x['@name'])

    # add numerical ids
    for i, annotation in enumerate(cvat_image_annotations):
        annotation['@id'] = i

    with open(IMAGES_DIR / 'CVAT/annotations.xml', 'w') as f:
        xmltodict.unparse(cvat_annotations, output=f, pretty=True)


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
