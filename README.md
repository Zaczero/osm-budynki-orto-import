# osm-budynki-orto-import

![Python version](https://shields.monicz.dev/github/pipenv/locked/python-version/Zaczero/osm-budynki-orto-import)
[![Project license](https://shields.monicz.dev/github/license/Zaczero/osm-budynki-orto-import)](https://github.com/Zaczero/osm-budynki-orto-import/blob/main/LICENSE)
[![Support my work](https://shields.monicz.dev/badge/%E2%99%A5%EF%B8%8F%20Support%20my%20work-purple)](https://monicz.dev/#support-my-work)
[![GitHub repo stars](https://shields.monicz.dev/github/stars/Zaczero/osm-budynki-orto-import?style=social)](https://github.com/Zaczero/osm-budynki-orto-import)

🧿 OpenStreetMap, AI import tool for buildings in Poland

## 💡 How it works

1. Fetches building data from [BDOT10k](https://bdot10k.geoportal.gov.pl/), convieniently provided by [budynki.openstreetmap.org.pl](https://budynki.openstreetmap.org.pl/).
2. Retrieves [ortophoto imagery](https://www.geoportal.gov.pl/dane/ortofotomapa) for each of the buildings and [preprocesses it](https://github.com/Zaczero/osm-budynki-orto-import/blob/main/processor.py).
3. Utilizes a fine-tuned [MobileNetV3Large model](https://github.com/Zaczero/osm-budynki-orto-import/blob/main/model.py) to classify the correctness of the BDOT10k information.
4. [Verifies historical OSM data](https://github.com/Zaczero/osm-budynki-orto-import/blob/main/overpass.py) to identify previously deleted buildings.
5. Imports new buildings into OSM.

## Reference

### Community discussion

https://community.openstreetmap.org/t/pomysl-automatyczny-import-budynkow-porownujac-obrysy-budynki-z-orto-geoportalu/100862/

### Data usage terms

https://www.geoportal.gov.pl/regulamin

https://wiki.openstreetmap.org/wiki/Pl:Geoportal.gov.pl

## Footer

### Contact me

https://monicz.dev/#get-in-touch

### Support my work

https://monicz.dev/#support-my-work

### License

This project is licensed under the GNU Affero General Public License v3.0.

The complete license text can be accessed in the repository at [LICENSE](https://github.com/Zaczero/osm-budynki-orto-import/blob/main/LICENSE).
