import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
from skimage import img_as_ubyte
from skimage.io import imsave

from config import SAVE_IMG, USER_AGENT


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


def save_image(image: np.ndarray, name: str = 'UNTITLED') -> None:
    if not SAVE_IMG:
        return

    if image.dtype in ('float32', 'float64', 'bool'):
        image = img_as_ubyte(image)

    imsave(f'images/{name}.png', image, check_contrast=False)


def meters_to_lat(meters: float) -> float:
    return meters / 111_111


def random_color() -> np.ndarray:
    color = np.random.rand(3)
    color = np.maximum(color, 0.2)
    return color
