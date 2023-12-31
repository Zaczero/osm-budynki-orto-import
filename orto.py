import httpx
import numpy as np
from skimage import img_as_float
from skimage.io import imread
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from utils import http_headers


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def fetch_orto(box: Box) -> np.ndarray:
    p1_lat, p1_lon = box.point
    p2_lat, p2_lon = box.point + box.size
    bbox = f'{p1_lat},{p1_lon},{p2_lat},{p2_lon}'

    r = httpx.get('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution', params={
        'LAYERS': 'Raster',
        'STYLES': 'default',
        'FORMAT': 'image/png',
        'CRS': 'EPSG:4326',
        'WIDTH': 256,
        'HEIGHT': 256,
        'BBOX': bbox,
        'VERSION': '1.3.0',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
    }, headers=http_headers(), timeout=60)

    r.raise_for_status()

    img = imread(r.content, plugin='imageio')

    if img.min() == img.max():
        raise Exception('Ortophoto imagery is not available')

    return img_as_float(img)
