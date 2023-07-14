import httpx
import numpy as np
from skimage.io import imread
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from utils import http_headers

PX_PER_LAT = 0.000002


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def fetch_orto(box: Box) -> tuple[np.ndarray]:
    p1 = box.point
    p2 = box.point + box.size
    bbox = f'{p1[0]},{p1[1]},{p2[0]},{p2[1]}'

    width = box.size[1] // PX_PER_LAT
    height = box.size[0] // PX_PER_LAT

    r = httpx.get('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution', params={
        'LAYERS': 'Raster',
        'STYLES': 'default',
        'FORMAT': 'image/png',
        'CRS': 'EPSG:4326',
        'WIDTH': width,
        'HEIGHT': height,
        'BBOX': bbox,
        'VERSION': '1.3.0',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
    }, headers=http_headers(), timeout=60)

    r.raise_for_status()

    return imread(r.content, as_gray=True, plugin='imageio')
