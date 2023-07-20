import json
import traceback
from multiprocessing import Pool
from pathlib import Path
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
    label: int
    data: dict | list
    image: np.ndarray

    def get_polygon(self) -> Polygon:
        return Polygon(tuple(LatLon(*p) for p in self.data[0]))


def _tag_to_label(tag: dict) -> int | None:
    if tag['@source'] != 'manual':
        print(f'[DATASET] ⚠️ Unknown source: {tag["@source"]!r}')
        return None

    if tag['@label'] in {'good'}:
        return 1
    elif tag['@label'] in {'bad', 'fundament'}:
        return 0
    elif tag['@label'] in {'unknown'}:
        return None

    raise ValueError(f'Unknown tag label: {tag["@label"]!r}')


def _iter_dataset(*, ignore_ids: frozenset[str]) -> Iterable[DatasetEntry]:
    cvat_annotation_paths = tuple(sorted(DATASET_DIR.rglob('annotations.xml')))

    for i, p in enumerate(cvat_annotation_paths, 1):
        dir_progress = f'{i}/{len(cvat_annotation_paths)}'
        cvat_dir = p.parent
        print(f'[DATASET][{dir_progress}] 📁 Iterating: {cvat_dir!r}')

        cvat_annotations = xmltodict.parse(p.read_text(), force_list=('image', 'tag', 'attribute'))
        cvat_annotations = cvat_annotations['annotations']['image']

        for j, annotation in enumerate(cvat_annotations, 1):
            file_progress = f'{j}/{len(cvat_annotations)}'
            raw_path = cvat_dir / Path(annotation['@name'])
            identifier = raw_path.stem

            if identifier in ignore_ids:
                continue

            print(f'[DATASET][{dir_progress}][{file_progress}] 📄 Iterating: {identifier!r}')

            if len(annotation['tag']) != 1:
                print(f'[DATASET] ⚠️ Unexpected tag count {len(annotation["tag"])} for {identifier!r}')
                continue

            label = _tag_to_label(annotation['tag'][0])

            if label is None:
                continue

            raw_name = raw_path.name
            raw_name_safe = raw_name.replace('.', '_')
            polygon_path = cvat_dir / 'images' / 'related_images' / raw_name_safe / 'polygon.json'

            data = json.loads(polygon_path.read_text())
            image = imread(raw_path)
            image = img_as_float(image)

            yield DatasetEntry(identifier, label, data, image)


def _process_building(building: Building) -> tuple[Building, dict | None]:
    with print_run_time('Process building'):
        try:
            return building, process_polygon(building.polygon, create_dataset=True)
        except:
            traceback.print_exc()
            return building, None


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


def _process_dataset_entry(entry: DatasetEntry) -> dict | None:
    polygon = entry.get_polygon()
    process_result = process_polygon(polygon, orto_rgb_img=entry.image)

    if process_result is None:
        return None

    process_result['id'] = entry.id
    process_result['label'] = entry.label
    return process_result


def process_dataset() -> None:
    if MODEL_DATASET_PATH.is_file():
        print(f'[DATASET] 💾 Loading existing dataset: {MODEL_DATASET_PATH}')
        df = pd.read_csv(MODEL_DATASET_PATH, index_col=False)
        processed_ids = frozenset(df['id'])
    else:
        df = pd.DataFrame()
        processed_ids = frozenset()

    result = []

    with Pool(CPU_COUNT) as pool:
        if CPU_COUNT == 1:
            iterator = map(_process_dataset_entry, _iter_dataset(ignore_ids=processed_ids))
        else:
            iterator = pool.imap_unordered(_process_dataset_entry, _iter_dataset(ignore_ids=processed_ids))

        for process_result in iterator:
            if process_result is None:
                continue

            result.append(process_result)

    df = pd.concat((df, pd.DataFrame(result)))
    df.to_csv(MODEL_DATASET_PATH, index=False)
