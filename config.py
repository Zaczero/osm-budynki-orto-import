import os
import secrets
from datetime import timedelta
from pathlib import Path

from pymongo import ASCENDING, IndexModel, MongoClient
from tinydb import TinyDB

from orjson_storage import ORJSONStorage

SEED = 42

SAVE_IMG = os.getenv('SAVE_IMG', '0') == '1'
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
SKIP_CONSTRUCTION = os.getenv('SKIP_CONSTRUCTION', '1') == '1'

DAILY_IMPORT_SPEED = float(os.getenv('DAILY_IMPORT_SPEED', '1000'))
SLEEP_AFTER_ONE_IMPORT = 86400 / DAILY_IMPORT_SPEED if DAILY_IMPORT_SPEED > 0 else 0
SLEEP_AFTER_GRID_ITER = timedelta(days=float(os.getenv('SLEEP_AFTER_GRID_ITER_DAYS', '30'))).total_seconds()
PROCESS_NICE = int(os.getenv('PROCESS_NICE', '15'))

if DRY_RUN:
    print('🦺 TEST MODE 🦺')
else:
    print('🔴 PRODUCTION MODE 🔴')

# Dedicated instance unavailable? Pick one from the public list:
# https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
OVERPASS_API_INTERPRETER = os.getenv('OVERPASS_API_INTERPRETER', 'https://overpass.monicz.dev/api/interpreter')

OSM_USERNAME = os.getenv('OSM_USERNAME')
OSM_PASSWORD = os.getenv('OSM_PASSWORD')
assert OSM_USERNAME and OSM_PASSWORD, 'OSM credentials not set'

CPU_COUNT = min(int(os.getenv('CPU_COUNT', '1')), len(os.sched_getaffinity(0)))

SCORER_VERSION = 3  # changing this will invalidate previous results

VERSION = '2.2.1'
CREATED_BY = f'osm-budynki-orto-import {VERSION}'
WEBSITE = 'https://github.com/Zaczero/osm-budynki-orto-import'
USER_AGENT = f'osm-budynki-orto-import/{VERSION} (+{WEBSITE})'

CHANGESET_ID_PLACEHOLDER = f'__CHANGESET_ID_PLACEHOLDER__{secrets.token_urlsafe(8)}__'

DEFAULT_CHANGESET_TAGS = {
    'comment': 'Import budynków z BDOT10k',
    'created_by': CREATED_BY,
    'import': 'yes',
    'source': 'budynki.openstreetmap.org.pl',
    'website': WEBSITE,
    'website:import': 'https://wiki.openstreetmap.org/wiki/BDOT10k_buildings_import',
}

MODEL_RESOLUTION = 224
PRECISION = 0.997
CONFIDENCE = 0.875

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

IMAGES_DIR = Path('images')
IMAGES_DIR.mkdir(exist_ok=True)

DATASET_DIR = Path('dataset')
DATASET_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path('model')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'model.h5'

DB_PATH = DATA_DIR / 'db.json'
DB = TinyDB(DB_PATH, storage=ORJSONStorage)
DB_GRID = DB.table('grid')

MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
MONGO = MongoClient(MONGO_URL)
MONGO_DB = MONGO['default']
MONGO_ADDED = MONGO_DB['added']

MONGO_ADDED.create_indexes([
    IndexModel([('unique_id', ASCENDING)])
])
