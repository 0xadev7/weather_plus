# Weather+ (Open-Meteo compatible, locally calibrated)

This service fetches a strong baseline (default: Open-Meteo) and applies **per-variable calibration** (EMOS/NGR / GBDT / QRF) to **out-perform Open-Meteo** on your box/time via hindcast-trained artifacts.

## Run

1. Setup Environment

```bash
conda create -n wxplus python=3.11 -y && conda activate wxplus
pip install -e . && pip install cdsapi pyarrow
```

2. CDS API

```bash
~/.cdsapirc
```

3. Fetch data & Train

```
bash scripts/global_fetch_and_train.sh
python weather_plus/scripts/train_tiles.py
```

4. Serve
   uvicorn weather_plus.api.server:app --host 0.0.0.0 --port 8000
