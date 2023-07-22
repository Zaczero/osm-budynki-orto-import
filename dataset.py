import json
import pickle
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import xmltodict
from scipy.ndimage import distance_transform_edt
from skimage import img_as_float, morphology, transform
from skimage.io import imread

from budynki import Building, fetch_buildings
from config import (CACHE_DIR, CPU_COUNT, DATASET_DIR, IMAGES_DIR,
                    MODEL_DATASET_PATH)
from db_grid import random_grid
from latlon import LatLon
from polygon import Polygon
from processor import ProcessResult, normalize_image, process_polygon
from utils import print_run_time, save_image


class DatasetEntry(NamedTuple):
    id: str
    label: int
    image: np.ndarray


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


def _apply_mask(image: np.ndarray, mask: np.ndarray, max_distance: float = 80) -> np.ndarray:
    mask_bold = morphology.dilation(mask, morphology.disk(2))

    # calculate the distance from each point in the mask to the nearest zero
    distances = distance_transform_edt(mask_bold < 0.5)

    # normalize the distances using your chosen max_distance,
    # so that they fall into the range [0, 1]
    normalized_distances = 1 - (distances / max_distance)
    normalized_distances = np.clip(normalized_distances, 0.1, 1)

    # multiply the image by the distance mask
    result = image * normalized_distances[..., np.newaxis]

    # add mask outline
    mask_outline = mask_bold - mask
    mask_outline = np.clip(mask_outline, 0, 0.6)
    mask_outline = mask_outline[..., np.newaxis]
    red_image = np.ones_like(image)
    red_image[..., 1:] = 0
    result = result * (1 - mask_outline) + red_image * mask_outline

    # save_image(result, force=True)

    return result


def _iter_dataset_identifier(identifier: str, raw_path: Path, annotation: dict, *, resolution: int = 224) -> DatasetEntry | None:
    cache_path = CACHE_DIR / f'DatasetEntry_{identifier}.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())

    if len(annotation['tag']) != 1:
        print(f'[DATASET] ⚠️ Unexpected tag count {len(annotation["tag"])} for {identifier!r}')
        return None

    label = _tag_to_label(annotation['tag'][0])

    if label is None:
        return None

    raw_name = raw_path.name
    raw_name_safe = raw_name.replace('.', '_')
    related_files_dir = raw_path.parent / 'related_images' / raw_name_safe

    mask_path = related_files_dir / 'mask.png'

    image = imread(raw_path)
    image = img_as_float(image)
    image = normalize_image(image)

    mask = imread(mask_path)[:, :, 0]
    mask = img_as_float(mask)

    result_image = _apply_mask(image, mask)
    result_image = transform.resize(result_image, (resolution, resolution), anti_aliasing=True)
    result_image = result_image * 2 - 1

    entry = DatasetEntry(identifier, label, result_image)
    cache_path.write_bytes(pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
    return entry


def iter_dataset(*, resolution: int = 224) -> Iterable[DatasetEntry]:
    cvat_annotation_paths = tuple(sorted(DATASET_DIR.rglob('annotations.xml')))
    done = set()

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
            if identifier in done:
                continue

            print(f'[DATASET][{dir_progress}][{file_progress}] 📄 Iterating: {identifier!r}')

            entry = _iter_dataset_identifier(identifier, raw_path, annotation, resolution=resolution)
            if entry is None:
                continue

            done.add(identifier)
            yield entry


def _process_building(building: Building) -> tuple[Building, ProcessResult | None]:
    with print_run_time('Process building'):
        try:
            return building, process_polygon(building.polygon)
        except:
            traceback.print_exc()
            return building, None


def create_dataset(size: int) -> None:
    # for p in DATASET_DIR.rglob('polygon.json'):
    #     for pp in p.parent.glob('*.png'):
    #         pp.unlink(missing_ok=True)

    #     unique_id = p.parent.name[:-4]
    #     polygon = Polygon(points=tuple(LatLon(*p) for p in json.loads(p.read_text())[0]))
    #     result = process_polygon(polygon)

    #     save_image(result.image, p.parent.parent.parent / unique_id, final_path=True, force=True)
    #     save_image(result.mask, p.parent / 'mask', final_path=True, force=True)
    #     save_image(result.overlay, p.parent / 'overlay', final_path=True, force=True)

    # with print_run_time('Loading model'):
    #     model = Model()

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
                buildings = fetch_buildings(cell)[:(size - len(cvat_image_annotations))]

            if not buildings:
                print('[CELL] ⏭️ Nothing to do')
                continue

            if CPU_COUNT == 1:
                iterator = map(_process_building, buildings)
            else:
                iterator = pool.imap_unordered(_process_building, buildings)

            for building, result in iterator:
                if result is None:
                    continue

                unique_id = building.polygon.unique_id_hash()

                # is_valid, proba = model.predict_single(data, threshold=0.5)
                # label = 'good' if is_valid else 'bad'

                raw_path = save_image(result.image, f'CVAT/images/{unique_id}', force=True)
                raw_name = raw_path.name
                raw_name_safe = raw_name.replace('.', '_')

                save_image(result.mask, f'CVAT/images/related_images/{raw_name_safe}/mask', force=True)
                save_image(result.overlay, f'CVAT/images/related_images/{raw_name_safe}/overlay', force=True)

                with open(IMAGES_DIR / f'CVAT/images/related_images/{raw_name_safe}/polygon.json', 'w') as f:
                    json.dump(building.polygon, f)

                annotation = {
                    '@name': f'images/{raw_name}',
                    '@height': result.image.shape[0],
                    '@width': result.image.shape[1],
                    # 'tag': [{
                    #     '@label': label,
                    #     '@source': 'auto',
                    #     'attribute': [{
                    #         '@name': 'proba',
                    #         '#text': f'{proba:.3f}'
                    #     }]
                    # }]
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
