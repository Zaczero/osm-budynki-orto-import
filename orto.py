import httpx
import numpy as np
from skimage.io import imread
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from utils import http_headers, meters_to_lat, meters_to_lon

_DENSITY = 0.14


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def fetch_orto(box: Box) -> tuple[np.ndarray]:
    p1 = box.point
    p2 = box.point + box.size
    bbox = f'{p1[0]},{p1[1]},{p2[0]},{p2[1]}'

    density_y = meters_to_lat(_DENSITY)
    density_x = meters_to_lon(_DENSITY, box.point[0])

    height = int(box.size[0] / density_y)
    width = int(box.size[1] / density_x)

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

    return imread(r.content, plugin='imageio')
