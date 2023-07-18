import os
import secrets
from pathlib import Path

from tinydb import TinyDB

SEED = 42

SAVE_IMG = os.getenv('SAVE_IMG', '0') == '1'
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
SLEEP_AFTER_IMPORT = float(os.getenv('SLEEP_AFTER_IMPORT', '1')) * 3600

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

SCORER_VERSION = 1  # changing this will invalidate previous results

VERSION = '1.0'
CREATED_BY = f'osm-budynki-orto-import {VERSION}'
WEBSITE = 'https://github.com/Zaczero/osm-budynki-orto-import'
USER_AGENT = f'osm-budynki-orto-import/{VERSION} (+{WEBSITE})'

CHANGESET_ID_PLACEHOLDER = f'__CHANGESET_ID_PLACEHOLDER__{secrets.token_urlsafe(8)}__'

DEFAULT_CHANGESET_TAGS = {
    'comment': 'Import budynków z ewidencji',
    'created_by': CREATED_BY,
    'import': 'yes',
    'source': 'budynki.openstreetmap.org.pl',
    'website': WEBSITE,
}

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

IMAGES_DIR = Path('images')
IMAGES_DIR.mkdir(exist_ok=True)

DATASET_DIR = Path('dataset')
DATASET_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / 'db.json'
DB = TinyDB(DB_PATH)
DB_ADDED = DB.table('added')
DB_GRID = DB.table('grid')
