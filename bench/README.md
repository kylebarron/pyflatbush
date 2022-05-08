# Bench setup

`pip install fiona`

```bash
> wget https://usbuildingdata.blob.core.windows.net/usbuildings-v2/NorthDakota.geojson.zip
> extract NorthDakota.geojson.zip
> fio cat NorthDakota.geojson/NorthDakota.geojson | fio bounds > bounds.txt
```

```bash
> wc -l bounds.txt
568213 bounds.txt
```

```bash
python bench.py
```

```bash
yarn install
yarn --silent node bench.mjs
```
