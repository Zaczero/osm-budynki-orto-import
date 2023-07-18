import time
from contextlib import contextmanager
from math import cos, pi, radians
from typing import Generator

import numpy as np
from skimage import img_as_ubyte
from skimage.io import imsave

from config import IMAGES_DIR, SAVE_IMG, USER_AGENT


@contextmanager
def print_run_time(message: str | list) -> Generator[None, None, None]:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # support message by reference
        if isinstance(message, list):
            message = message[0]

        print(f'[⏱️] {message} took {elapsed_time:.3f}s')


def http_headers() -> dict:
    return {
        'User-Agent': USER_AGENT,
    }


def save_image(image: np.ndarray, name: str = 'UNTITLED', *, force: bool = False) -> None:
    if not SAVE_IMG and not force:
        return

    if image.dtype in ('float32', 'float64', 'bool'):
        image = img_as_ubyte(image)

    imsave(IMAGES_DIR / f'{name}.png', image, check_contrast=False)


def random_color() -> np.ndarray:
    color = np.random.rand(3)
    color = np.maximum(color, 0.2)
    return color


EARTH_RADIUS = 6371000
CIRCUMFERENCE = 2 * pi * EARTH_RADIUS


def meters_to_lat(meters: float) -> float:
    return meters / (CIRCUMFERENCE / 360)


def meters_to_lon(meters: float, lat: float) -> float:
    return meters / ((CIRCUMFERENCE / 360) * cos(radians(lat)))
